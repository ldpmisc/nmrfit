from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np

from nmrFit_v0 import ParamRef, Peak


class ConstraintType(Enum):
    """Enumeration of supported constraint types."""
    LINEAR = "LINEAR"
    RELAX_EXP = "RELAX_EXP"


@dataclass
class ConstraintValidationError:
    """Container for constraint validation errors."""
    target_pref: ParamRef
    message: str


class ConstraintRule(ABC):
    """
    Abstract base class for all constraint rules.
    
    A constraint rule defines a mathematical relationship between a target parameter
    and optionally a driver parameter. Rules can be validated, evaluated, and applied
    to lmfit Parameter objects.
    """
    
    enabled: bool = True
    
    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType:
        """Return the type of this constraint."""
        pass
    
    @abstractmethod
    def validate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> List[str]:
        """
        Validate constraint preconditions.
        
        Args:
            target_peak: Peak object for target parameter
            target_pref: ParamRef for target parameter
            driver_peak: Peak object for driver parameter (if any)
            driver_pref: ParamRef for driver parameter (if any)
            slice_states: Dict mapping slice_id → SliceFitState
        
        Returns:
            List of error messages (empty if valid)
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> float:
        """
        Evaluate the constraint to compute target value.
        
        Args:
            target_peak: Peak object for target parameter
            target_pref: ParamRef for target parameter
            driver_peak: Peak object for driver parameter (if any)
            driver_pref: ParamRef for driver parameter (if any)
            slice_states: Dict mapping slice_id → SliceFitState
        
        Returns:
            Computed value for target parameter in its native units
        """
        pass
    
    @abstractmethod
    def apply_to_lmfit(
        self,
        params,
        target_pref: ParamRef,
        evaluated_value: float
    ) -> None:
        """
        Apply constraint to an lmfit Parameters object.
        
        Args:
            params: lmfit.Parameters instance
            target_pref: ParamRef for target parameter
            evaluated_value: Pre-computed value from evaluate()
        """
        pass


class LinearConstraint(ConstraintRule):
    """
    Linear constraint: target = a * driver + b
    
    If driver_pref is None, the constraint is a fixed value: target = b
    """
    
    def __init__(
        self,
        target_pref: ParamRef,
        driver_pref: Optional[ParamRef] = None,
        a: float = 1.0,
        b: float = 0.0,
        enabled: bool = True
    ):
        """
        Initialize a linear constraint.
        
        Args:
            target_pref: Parameter being constrained
            driver_pref: Parameter driving the constraint (optional)
            a: Multiplicative coefficient
            b: Additive offset
            enabled: Whether this constraint is active
        """
        self.target_pref = target_pref
        self.driver_pref = driver_pref
        self.a = float(a)
        self.b = float(b)
        self.enabled = bool(enabled)
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.LINEAR
    
    def validate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> List[str]:
        """Validate that driver exists if required."""
        errors = []
        
        if self.driver_pref is not None and driver_peak is None:
            errors.append(
                f"Linear constraint on {self._fmt_pref(target_pref)}: "
                f"driver {self._fmt_pref(self.driver_pref)} not found"
            )
        
        # Validate coefficients
        if not np.isfinite(self.a):
            errors.append(
                f"Linear constraint on {self._fmt_pref(target_pref)}: "
                f"coefficient a={self.a} is not finite"
            )
        
        if not np.isfinite(self.b):
            errors.append(
                f"Linear constraint on {self._fmt_pref(target_pref)}: "
                f"offset b={self.b} is not finite"
            )
        
        return errors
    
    def evaluate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> float:
        """
        Evaluate: target = a * driver_value + b
        If no driver, return b (fixed value).
        """
        if self.driver_pref is None or driver_peak is None:
            return float(self.b)
        
        # Extract driver parameter value from driver_peak
        driver_value = self._get_peak_value(driver_peak, self.driver_pref.name)
        
        result = self.a * driver_value + self.b
        return float(result)
    
    def apply_to_lmfit(
        self,
        params,
        target_pref: ParamRef,
        evaluated_value: float
    ) -> None:
        """
        Apply linear constraint to lmfit.
        
        For LINEAR constraints with a driver: set vary=False and expr='...'
        For fixed values (no driver): set value and vary=False
        """
        param_key = self._pref_to_lmfit_key(target_pref)
        
        if param_key not in params:
            return
        
        param = params[param_key]
        
        if self.driver_pref is None:
            # Fixed value constraint
            param.value = evaluated_value
            param.vary = False
        else:
            # Expression-based constraint: build expression string
            driver_key = self._pref_to_lmfit_key(self.driver_pref)
            
            # Build expression: a*driver_key + b
            if self.a == 1.0 and self.b == 0.0:
                expr = driver_key
            elif self.a == 1.0:
                expr = f"{driver_key} + {self.b}"
            elif self.b == 0.0:
                expr = f"{self.a} * {driver_key}"
            else:
                expr = f"{self.a} * {driver_key} + {self.b}"
            
            param.expr = expr
            param.vary = False
    
    @staticmethod
    def _get_peak_value(peak: Peak, param_name: str) -> float:
        """Extract parameter value from Peak object."""
        param_name = str(param_name).lower().strip()
        
        if param_name == "pos":
            return float(peak.pos)
        elif param_name == "amp":
            return float(peak.amp)
        elif param_name == "lor":
            return float(peak.lor_hz)
        elif param_name == "gauss":
            return float(peak.gauss_disp)
        else:
            raise ValueError(f"Unknown peak parameter: {param_name}")
    
    @staticmethod
    def _pref_to_lmfit_key(pref: ParamRef) -> str:
        """Convert ParamRef to lmfit parameter key."""
        return f"s{int(pref.slice_id)}_p{int(pref.peak_id)}_{str(pref.name)}"
    
    @staticmethod
    def _fmt_pref(pref: ParamRef) -> str:
        """Format ParamRef for display."""
        return f"s{pref.slice_id}_p{pref.peak_id}_{pref.name}"


class RelaxExponentialConstraint(ConstraintRule):
    """
    Relaxation exponential constraint: target = driver * A * exp(-t / T) + C
    
    Used for fitting relaxation curves where target values depend on acquisition time.
    Time can be supplied explicitly or read from a time registry.
    """
    
    def __init__(
        self,
        target_pref: ParamRef,
        driver_pref: ParamRef,
        T_seconds: float,
        A: float = 1.0,
        C: float = 0.0,
        time_seconds: Optional[float] = None,
        T_name: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize a relaxation exponential constraint.
        
        Args:
            target_pref: Parameter being constrained
            driver_pref: Parameter driving the decay
            T_seconds: Decay time constant (seconds)
            A: Amplitude multiplicative factor
            C: Baseline offset
            time_seconds: Explicit time value (overrides T_name lookup)
            T_name: Name of time constant in seed registry (optional)
            enabled: Whether this constraint is active
        """
        self.target_pref = target_pref
        self.driver_pref = driver_pref
        self.T_seconds = float(T_seconds)
        self.A = float(A)
        self.C = float(C)
        self.time_seconds = float(time_seconds) if time_seconds is not None else None
        self.T_name = T_name
        self.enabled = bool(enabled)
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.RELAX_EXP
    
    def validate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> List[str]:
        """Validate exponential constraint parameters."""
        errors = []
        
        if driver_peak is None or driver_pref is None:
            errors.append(
                f"Exponential constraint on {self._fmt_pref(target_pref)}: "
                f"driver {self._fmt_pref(self.driver_pref)} not found"
            )
        
        # Validate T
        if self.T_seconds <= 0.0:
            errors.append(
                f"Exponential constraint on {self._fmt_pref(target_pref)}: "
                f"time constant T={self.T_seconds} must be positive"
            )
        
        # Validate time
        if self.time_seconds is None and self.T_name is None:
            errors.append(
                f"Exponential constraint on {self._fmt_pref(target_pref)}: "
                f"no time_seconds or T_name provided"
            )
        
        if self.time_seconds is not None and self.time_seconds < 0.0:
            errors.append(
                f"Exponential constraint on {self._fmt_pref(target_pref)}: "
                f"time_seconds={self.time_seconds} must be non-negative"
            )
        
        # Validate coefficients
        if not np.isfinite(self.A):
            errors.append(
                f"Exponential constraint on {self._fmt_pref(target_pref)}: "
                f"amplitude A={self.A} is not finite"
            )
        
        if not np.isfinite(self.C):
            errors.append(
                f"Exponential constraint on {self._fmt_pref(target_pref)}: "
                f"baseline C={self.C} is not finite"
            )
        
        return errors
    
    def evaluate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> float:
        """
        Evaluate: target = driver * A * exp(-t / T) + C
        """
        if driver_peak is None or driver_pref is None:
            raise ValueError("Driver peak required for exponential constraint")
        
        # Get time value
        t = self._get_time_value(slice_states)
        
        # Extract driver value
        driver_value = LinearConstraint._get_peak_value(driver_peak, self.driver_pref.name)
        
        # Compute: driver * A * exp(-t / T) + C
        exp_term = np.exp(-t / self.T_seconds)
        result = driver_value * self.A * exp_term + self.C
        
        return float(result)
    
    def apply_to_lmfit(
        self,
        params,
        target_pref: ParamRef,
        evaluated_value: float
    ) -> None:
        """
        Apply exponential constraint to lmfit.
        
        For exponential constraints, set the parameter to fixed value
        (expression building would be complex; evaluation is done once upfront).
        """
        param_key = LinearConstraint._pref_to_lmfit_key(target_pref)
        
        if param_key not in params:
            return
        
        param = params[param_key]
        param.value = evaluated_value
        param.vary = False
    
    def _get_time_value(self, slice_states: Dict[int, Any]) -> float:
        """Get time value from explicit value or from seed registry."""
        if self.time_seconds is not None:
            return float(self.time_seconds)
        
        if self.T_name is not None:
            # Would lookup from MainWindow._TSeedRegistry or similar
            # For now, raise to indicate missing implementation
            raise NotImplementedError(
                f"T_name lookup for '{self.T_name}' not yet implemented; "
                f"use explicit time_seconds instead"
            )
        
        raise ValueError("No time value available for exponential constraint")
    
    @staticmethod
    def _fmt_pref(pref: ParamRef) -> str:
        """Format ParamRef for display."""
        return f"s{pref.slice_id}_p{pref.peak_id}_{pref.name}"


class ConstraintRuleFactory:
    """
    Factory for creating constraint rules from serialized representations.
    Useful for loading/saving constraints to file.
    """
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ConstraintRule:
        """
        Create a ConstraintRule from a dictionary representation.
        
        Expected keys:
            type: "LINEAR" or "RELAX_EXP"
            target_pref: {slice_id, peak_id, name}
            [driver_pref]: {slice_id, peak_id, name}  (optional for LINEAR)
            [a, b]: for LINEAR
            [T_seconds, A, C, time_seconds, T_name]: for RELAX_EXP
            [enabled]: True/False
        """
        constraint_type = str(data.get("type", "LINEAR")).upper()
        enabled = data.get("enabled", True)
        
        # Parse target
        target_data = data.get("target_pref", {})
        target_pref = ParamRef(
            slice_id=int(target_data["slice_id"]),
            peak_id=int(target_data["peak_id"]),
            name=str(target_data["name"])
        )
        
        if constraint_type == "LINEAR":
            driver_data = data.get("driver_pref")
            driver_pref = None
            if driver_data is not None:
                driver_pref = ParamRef(
                    slice_id=int(driver_data["slice_id"]),
                    peak_id=int(driver_data["peak_id"]),
                    name=str(driver_data["name"])
                )
            
            return LinearConstraint(
                target_pref=target_pref,
                driver_pref=driver_pref,
                a=float(data.get("a", 1.0)),
                b=float(data.get("b", 0.0)),
                enabled=enabled
            )
        
        elif constraint_type == "RELAX_EXP":
            driver_data = data.get("driver_pref")
            driver_pref = ParamRef(
                slice_id=int(driver_data["slice_id"]),
                peak_id=int(driver_data["peak_id"]),
                name=str(driver_data["name"])
            )
            
            return RelaxExponentialConstraint(
                target_pref=target_pref,
                driver_pref=driver_pref,
                T_seconds=float(data.get("T_seconds", 1.0)),
                A=float(data.get("A", 1.0)),
                C=float(data.get("C", 0.0)),
                time_seconds=data.get("time_seconds"),
                T_name=data.get("T_name"),
                enabled=enabled
            )
        
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
    
    @staticmethod
    def to_dict(rule: ConstraintRule) -> Dict[str, Any]:
        """Serialize a ConstraintRule to a dictionary."""
        result = {
            "type": rule.constraint_type.value,
            "enabled": rule.enabled,
            "target_pref": {
                "slice_id": rule.target_pref.slice_id,
                "peak_id": rule.target_pref.peak_id,
                "name": rule.target_pref.name,
            },
        }
        
        if isinstance(rule, LinearConstraint):
            if rule.driver_pref is not None:
                result["driver_pref"] = {
                    "slice_id": rule.driver_pref.slice_id,
                    "peak_id": rule.driver_pref.peak_id,
                    "name": rule.driver_pref.name,
                }
            result["a"] = rule.a
            result["b"] = rule.b
        
        elif isinstance(rule, RelaxExponentialConstraint):
            result["driver_pref"] = {
                "slice_id": rule.driver_pref.slice_id,
                "peak_id": rule.driver_pref.peak_id,
                "name": rule.driver_pref.name,
            }
            result["T_seconds"] = rule.T_seconds
            result["A"] = rule.A
            result["C"] = rule.C
            if rule.time_seconds is not None:
                result["time_seconds"] = rule.time_seconds
            if rule.T_name is not None:
                result["T_name"] = rule.T_name
        
        return result