from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
import numpy as np

# Use TYPE_CHECKING to avoid circular import at runtime
# nmrFit_v0 → constraint_rules, constraint_rules → nmrFit_v0 creates cycle
if TYPE_CHECKING:
    from nmrFit_v0 import ParamRef, Peak, LinkExpr, LinkType


class ConstraintType(Enum):
    """Enumeration of supported constraint types."""
    LINEAR = "LINEAR"
    RELAX_EXP = "RELAX_EXP"


@dataclass
class ConstraintValidationError:
    """Container for constraint validation errors."""
    target_pref: Any  # ParamRef (deferred to avoid circular import)
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


class RelaxDecayConstraint(ConstraintRule):
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
        # Import at function level to avoid circular dependency
        from nmrFit_v0 import ParamRef as _ParamRef
        
        constraint_type = str(data.get("type", "LINEAR")).upper()
        enabled = data.get("enabled", True)
        
        # Parse target
        target_data = data.get("target_pref", {})
        target_pref = _ParamRef(
            slice_id=int(target_data["slice_id"]),
            peak_id=int(target_data["peak_id"]),
            name=str(target_data["name"])
        )
        
        if constraint_type == "LINEAR":
            driver_data = data.get("driver_pref")
            driver_pref = None
            if driver_data is not None:
                driver_pref = _ParamRef(
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
            driver_pref = _ParamRef(
                slice_id=int(driver_data["slice_id"]),
                peak_id=int(driver_data["peak_id"]),
                name=str(driver_data["name"])
            )
            
            return RelaxDecayConstraint(
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
        
        elif isinstance(rule, RelaxDecayConstraint):
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


# ============================================================================
# Migration Bridge: LinkExpr → ConstraintRule
# ============================================================================

def LinkExpr_to_ConstraintRule(link_expr) -> ConstraintRule:
    """
    Convert old LinkExpr to new ConstraintRule subclass.
    
    This function enables backward compatibility when loading existing fit files
    that use the deprecated LinkExpr format. It maps:
      - LinkExpr(type=LINEAR) → LinearConstraint
      - LinkExpr(type=RELAX_EXP) → RelaxDecayConstraint
    
    Args:
        link_expr: LinkExpr instance with type, target, driver, args, enabled
        
    Returns:
        ConstraintRule subclass instance (LinearConstraint or RelaxDecayConstraint)
        
    Raises:
        ValueError: if link_expr.type is unknown
    """
    # Import at function level to avoid circular dependency
    from nmrFit_v0 import LinkType as _LinkType
    
    if link_expr.type == _LinkType.LINEAR:
        return LinearConstraint(
            target_pref=link_expr.target,
            driver_pref=link_expr.driver,
            a=link_expr.args.get("a", 1.0),
            b=link_expr.args.get("b", 0.0),
            enabled=link_expr.enabled
        )
    
    elif link_expr.type == _LinkType.RELAX_EXP:
        return RelaxDecayConstraint(
            target_pref=link_expr.target,
            driver_pref=link_expr.driver,
            T_seconds=link_expr.args.get("T", 1.0),
            A=link_expr.args.get("A", 1.0),
            C=link_expr.args.get("C", 0.0),
            time_seconds=link_expr.args.get("t_override"),
            T_name=link_expr.args.get("T_name"),
            enabled=link_expr.enabled
        )
    
    else:
        raise ValueError(f"Unknown link type in LinkExpr: {link_expr.type}")


def ConstraintRule_to_LinkExpr(rule: ConstraintRule):
    """
    Convert new ConstraintRule to old LinkExpr format.
    
    This function enables backward compatibility when exporting to the old
    LinkExpr format. It is the inverse of LinkExpr_to_ConstraintRule.
    
    Args:
        rule: ConstraintRule subclass instance
        
    Returns:
        LinkExpr instance with type, target, driver, args, enabled
        
    Raises:
        ValueError: if rule type is unknown
    """
    # Import at function level to avoid circular dependency
    from nmrFit_v0 import LinkExpr as _LinkExpr, LinkType as _LinkType
    
    if isinstance(rule, LinearConstraint):
        return _LinkExpr(
            type=_LinkType.LINEAR,
            target=rule.target_pref,
            driver=rule.driver_pref,
            args={"a": rule.a, "b": rule.b},
            enabled=rule.enabled
        )
    
    elif isinstance(rule, RelaxDecayConstraint):
        args = {
            "T": rule.T_seconds,
            "A": rule.A,
            "C": rule.C
        }
        if rule.time_seconds is not None:
            args["t_override"] = rule.time_seconds
        if rule.T_name is not None:
            args["T_name"] = rule.T_name
        
        return _LinkExpr(
            type=_LinkType.RELAX_EXP,
            target=rule.target_pref,
            driver=rule.driver_pref,
            args=args,
            enabled=rule.enabled
        )
    
    else:
        raise ValueError(f"Unknown constraint rule type: {type(rule)}")


# ============================================================================
# ConstrainedPeak Aggregate: Wraps Peak + Constraints Dictionary
# ============================================================================

@dataclass
class ConstrainedPeak:
    """
    Aggregate that combines a Peak with its associated constraints.
    
    This is the core entity in the Aggregate Pattern for constraint management.
    Each ConstrainedPeak manages:
      1. A single Peak (amplitude, position, widths)
      2. A dictionary of ConstraintRules keyed by parameter name
      3. Validation and application of constraints to lmfit
    
    Attributes:
        peak: The Peak object (pos, amp, lor_hz, gauss_disp)
        constraints: Dict[str, ConstraintRule] mapping param names to rules
                     Example: {"amp": LinearConstraint(...), "pos": RelaxDecayConstraint(...)}
    """
    peak: Any  # Peak (deferred type to avoid circular import)
    constraints: Dict[str, ConstraintRule] = field(default_factory=dict)
    
    def get_constraint(self, param_name: str) -> Optional[ConstraintRule]:
        """
        Retrieve a constraint for a given parameter name.
        
        Args:
            param_name: Name of parameter ("pos", "amp", "lor", "gauss")
            
        Returns:
            ConstraintRule if one is registered, else None
        """
        return self.constraints.get(str(param_name).lower(), None)
    
    def set_constraint(self, param_name: str, rule: Optional[ConstraintRule]) -> None:
        """
        Register or unregister a constraint for a parameter.
        
        Args:
            param_name: Name of parameter ("pos", "amp", "lor", "gauss")
            rule: ConstraintRule to register, or None to clear
        """
        pname = str(param_name).lower()
        if rule is None:
            self.constraints.pop(pname, None)
        else:
            self.constraints[pname] = rule
    
    def validate(
        self,
        target_pref: Any,  # ParamRef (deferred type)
        driver_peaks: Dict[tuple, Any],  # {(slice_id, peak_id): Peak}
        slice_states: Dict[int, Any]  # {slice_id: SliceFitState}
    ) -> List[ConstraintValidationError]:
        """
        Validate all constraints for this peak.
        
        Args:
            target_pref: ParamRef identifying this peak's slice and peak_id
            driver_peaks: Dict mapping (slice_id, peak_id) → Peak for all drivers
            slice_states: Dict mapping slice_id → SliceFitState
            
        Returns:
            List of ConstraintValidationError for any validation failures (empty if valid)
        """
        errors: List[ConstraintValidationError] = []
        
        for param_name, rule in self.constraints.items():
            if rule is None or not getattr(rule, 'enabled', True):
                continue
            
            # Find driver peak if needed
            driver_pref = getattr(rule, 'driver_pref', None)
            driver_peak = None
            if driver_pref is not None:
                driver_key = (int(driver_pref.slice_id), int(driver_pref.peak_id))
                driver_peak = driver_peaks.get(driver_key, None)
            
            # Validate constraint
            try:
                validation_errors = rule.validate(
                    target_peak=self.peak,
                    target_pref=target_pref,
                    driver_peak=driver_peak,
                    driver_pref=driver_pref,
                    slice_states=slice_states
                )
                
                for error_msg in validation_errors:
                    errors.append(ConstraintValidationError(
                        target_pref=target_pref,
                        message=f"Parameter '{param_name}': {error_msg}"
                    ))
            except Exception as e:
                errors.append(ConstraintValidationError(
                    target_pref=target_pref,
                    message=f"Parameter '{param_name}': Validation exception: {str(e)}"
                ))
        
        return errors
    
    def apply_constraints_to_lmfit(
        self,
        params: Any,  # lmfit.Parameters
        target_pref: Any,  # ParamRef
        driver_peaks: Dict[tuple, Any],  # {(slice_id, peak_id): Peak}
        slice_states: Dict[int, Any],  # {slice_id: SliceFitState}
        name_map: Optional[Dict[str, str]] = None  # {normalized_name: lmfit_base_name}
    ) -> None:
        """
        Apply all enabled constraints to an lmfit Parameters object.
        
        This method:
          1. Evaluates each constraint to compute target values
          2. Sets lmfit Parameter.expr (for linear same-slice) or .value/.vary (for others)
          3. Handles cross-slice drivers by freezing with numeric values
        
        Args:
            params: lmfit.Parameters object to modify
            target_pref: ParamRef identifying this peak (slice_id, peak_id)
            driver_peaks: Dict {(slice_id, peak_id): Peak}
            slice_states: Dict {slice_id: SliceFitState}
            name_map: Optional dict mapping normalized param names to lmfit base names
                      Default: {"pos": "pos", "amp": "amp", "lor": "lor", "gauss": "gauss"}
        """
        if name_map is None:
            name_map = {"pos": "pos", "amp": "amp", "lor": "lor", "gauss": "gauss"}
        
        def _norm(n: str) -> str:
            return (n or "").strip().lower()
        
        for param_name, rule in self.constraints.items():
            if rule is None or not getattr(rule, 'enabled', True):
                continue
            
            # Resolve lmfit parameter name
            base_name = name_map.get(_norm(param_name))
            if not base_name:
                continue
            
            peak_id = int(target_pref.peak_id)
            key_tgt = f"{base_name}_{peak_id}"  # e.g., "amp_0", "pos_1"
            p_tgt = params.get(key_tgt, None)
            if p_tgt is None:
                continue
            
            # Find driver peak
            driver_pref = getattr(rule, 'driver_pref', None)
            driver_peak = None
            if driver_pref is not None:
                driver_key = (int(driver_pref.slice_id), int(driver_pref.peak_id))
                driver_peak = driver_peaks.get(driver_key, None)
            
            # Evaluate constraint to get target value
            try:
                evaluated_value = rule.evaluate(
                    target_peak=self.peak,
                    target_pref=target_pref,
                    driver_peak=driver_peak,
                    driver_pref=driver_pref,
                    slice_states=slice_states
                )
            except Exception:
                # If evaluation fails, skip this constraint
                continue
            
            # Apply to lmfit based on constraint type and driver location
            target_slice = int(target_pref.slice_id)
            
            # For LinearConstraint with same-slice driver: use algebraic expression
            if (isinstance(rule, LinearConstraint) and 
                driver_pref is not None and 
                int(driver_pref.slice_id) == target_slice):
                
                driver_base = name_map.get(_norm(driver_pref.name))
                if driver_base:
                    driver_peak_id = int(driver_pref.peak_id)
                    key_drv = f"{driver_base}_{driver_peak_id}"
                    
                    if key_drv in params:
                        a = float(getattr(rule, 'a', 1.0))
                        b = float(getattr(rule, 'b', 0.0))
                        expr = f"{key_drv}*{a}+{b}"
                        try:
                            p_tgt.set(value=float(evaluated_value), expr=expr)
                        except Exception:
                            p_tgt.set(value=float(evaluated_value), vary=False)
                        continue
            
            # All other cases: numeric freeze
            try:
                p_tgt.set(value=float(evaluated_value), vary=False)
            except Exception:
                pass


@dataclass
class ConstraintStore:
    """
    Per-slice registry for constraints indexed by ParamRef.
    
    ConstraintStore manages all constraints for a single slice, providing:
      - Registry: Dict[ParamRef, ConstraintRule]
      - Validation: validate_all() method with error collection
      - Application: apply_to_lmfit() for coordinated constraint application
      - Backward compatibility: from_LinkStore() converter
    
    The store maintains a reverse index for efficient dependent lookup during
    constraint application.
    """
    
    constraints: Dict[Any, ConstraintRule] = field(default_factory=dict)  # {ParamRef → ConstraintRule}
    reverse_index: Dict[Any, List[Any]] = field(default_factory=dict)  # {ParamRef(driver) → [ParamRef(targets)]}
    
    def __post_init__(self):
        """Rebuild reverse index after initialization."""
        self._rebuild_reverse_index()
    
    def _rebuild_reverse_index(self) -> None:
        """Rebuild the reverse index mapping drivers to targets."""
        self.reverse_index.clear()
        for target_pref, rule in self.constraints.items():
            if rule is None or not getattr(rule, 'enabled', True):
                continue
            driver_pref = getattr(rule, 'driver_pref', None)
            if driver_pref is not None:
                if driver_pref not in self.reverse_index:
                    self.reverse_index[driver_pref] = []
                self.reverse_index[driver_pref].append(target_pref)
    
    def add_constraint(self, target_pref: Any, rule: Optional[ConstraintRule]) -> None:
        """
        Register or unregister a constraint.
        
        Args:
            target_pref: ParamRef identifying the target parameter
            rule: ConstraintRule to apply, or None to unregister
        """
        if rule is None:
            self.constraints.pop(target_pref, None)
        else:
            self.constraints[target_pref] = rule
        self._rebuild_reverse_index()
    
    def get_constraint(self, target_pref: Any) -> Optional[ConstraintRule]:
        """Retrieve constraint for a target ParamRef."""
        return self.constraints.get(target_pref, None)
    
    def get_dependents(self, driver_pref: Any) -> List[Any]:
        """
        Get all target ParamRefs that depend on a driver.
        
        Args:
            driver_pref: ParamRef of driver parameter
        
        Returns:
            List of ParamRef objects for targets (empty if no dependents)
        """
        return self.reverse_index.get(driver_pref, [])
    
    def validate_all(
        self,
        peak_map: Dict[tuple, Any],  # {(slice_id, peak_id): Peak}
        slice_states: Dict[int, Any]  # {slice_id: SliceFitState}
    ) -> List[ConstraintValidationError]:
        """
        Validate all constraints in the store.
        
        Args:
            peak_map: Dict {(slice_id, peak_id): Peak}
            slice_states: Dict {slice_id: SliceFitState}
        
        Returns:
            List of ConstraintValidationError (empty if all valid)
        """
        errors = []
        
        for target_pref, rule in self.constraints.items():
            if rule is None or not getattr(rule, 'enabled', True):
                continue
            
            # Resolve target peak
            target_key = (int(target_pref.slice_id), int(target_pref.peak_id))
            target_peak = peak_map.get(target_key, None)
            if target_peak is None:
                errors.append(ConstraintValidationError(
                    target_pref=target_pref,
                    message=f"Target peak not found: {target_key}"
                ))
                continue
            
            # Resolve driver peak (if any)
            driver_pref = getattr(rule, 'driver_pref', None)
            driver_peak = None
            if driver_pref is not None:
                driver_key = (int(driver_pref.slice_id), int(driver_pref.peak_id))
                driver_peak = peak_map.get(driver_key, None)
                if driver_peak is None:
                    errors.append(ConstraintValidationError(
                        target_pref=target_pref,
                        message=f"Driver peak not found: {driver_key}"
                    ))
                    continue
            
            # Validate constraint
            try:
                error_msgs = rule.validate(
                    target_peak=target_peak,
                    target_pref=target_pref,
                    driver_peak=driver_peak,
                    driver_pref=driver_pref,
                    slice_states=slice_states
                )
                for msg in error_msgs:
                    errors.append(ConstraintValidationError(
                        target_pref=target_pref,
                        message=msg
                    ))
            except Exception as e:
                errors.append(ConstraintValidationError(
                    target_pref=target_pref,
                    message=f"Validation exception: {str(e)}"
                ))
        
        return errors
    
    def apply_to_lmfit(
        self,
        params,
        peak_map: Dict[tuple, Any],  # {(slice_id, peak_id): Peak}
        slice_states: Dict[int, Any],  # {slice_id: SliceFitState}
        name_map: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Apply all constraints to an lmfit Parameters object.
        
        Args:
            params: lmfit.Parameters object to modify
            peak_map: Dict {(slice_id, peak_id): Peak}
            slice_states: Dict {slice_id: SliceFitState}
            name_map: Optional dict mapping normalized param names to lmfit base names
        """
        if name_map is None:
            name_map = {"pos": "pos", "amp": "amp", "lor": "lor", "gauss": "gauss"}
        
        def _norm(n: str) -> str:
            return (n or "").strip().lower()
        
        for target_pref, rule in self.constraints.items():
            if rule is None or not getattr(rule, 'enabled', True):
                continue
            
            # Resolve target peak
            target_key = (int(target_pref.slice_id), int(target_pref.peak_id))
            target_peak = peak_map.get(target_key, None)
            if target_peak is None:
                continue
            
            # Resolve driver peak (if any)
            driver_pref = getattr(rule, 'driver_pref', None)
            driver_peak = None
            if driver_pref is not None:
                driver_key = (int(driver_pref.slice_id), int(driver_pref.peak_id))
                driver_peak = peak_map.get(driver_key, None)
            
            # Resolve lmfit parameter name
            base_name = name_map.get(_norm(target_pref.name))
            if not base_name:
                continue
            
            peak_id = int(target_pref.peak_id)
            key_tgt = f"{base_name}_{peak_id}"
            p_tgt = params.get(key_tgt, None)
            if p_tgt is None:
                continue
            
            # Evaluate constraint
            try:
                evaluated_value = rule.evaluate(
                    target_peak=target_peak,
                    target_pref=target_pref,
                    driver_peak=driver_peak,
                    driver_pref=driver_pref,
                    slice_states=slice_states
                )
            except Exception:
                continue
            
            # Apply to lmfit based on constraint type and driver location
            target_slice = int(target_pref.slice_id)
            
            # For LinearConstraint with same-slice driver: use algebraic expression
            if (isinstance(rule, LinearConstraint) and 
                driver_pref is not None and 
                int(driver_pref.slice_id) == target_slice):
                
                driver_base = name_map.get(_norm(driver_pref.name))
                if driver_base:
                    driver_peak_id = int(driver_pref.peak_id)
                    key_drv = f"{driver_base}_{driver_peak_id}"
                    
                    if key_drv in params:
                        a = float(getattr(rule, 'a', 1.0))
                        b = float(getattr(rule, 'b', 0.0))
                        expr = f"{key_drv}*{a}+{b}"
                        try:
                            p_tgt.set(value=float(evaluated_value), expr=expr)
                        except Exception:
                            p_tgt.set(value=float(evaluated_value), vary=False)
                        continue
            
            # All other cases: numeric freeze
            try:
                p_tgt.set(value=float(evaluated_value), vary=False)
            except Exception:
                pass
    
    @staticmethod
    def from_LinkStore(link_store: Any) -> ConstraintStore:
        """
        Create ConstraintStore from legacy LinkStore (backward compatibility).
        
        Args:
            link_store: LinkStore object with links list
        
        Returns:
            ConstraintStore with constraints converted from LinkExpr objects
        """
        store = ConstraintStore()
        
        if link_store is None or not hasattr(link_store, 'links'):
            return store
        
        for link_expr in link_store.links:
            if link_expr is None:
                continue
            try:
                rule = LinkExpr_to_ConstraintRule(link_expr)
                if rule is not None:
                    target_pref = getattr(link_expr, 'target', None)
                    if target_pref is not None:
                        store.add_constraint(target_pref, rule)
            except Exception:
                # Skip malformed LinkExpr objects
                continue
        
        return store
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize ConstraintStore to JSON-compatible dict.
        
        Returns:
            Dict with 'constraints' key containing serialized rules
        """
        constraints_dict = {}
        for pref, rule in self.constraints.items():
            pref_key = f"s{pref.slice_id}_p{pref.peak_id}_{pref.name}"
            if rule is not None:
                constraints_dict[pref_key] = ConstraintRuleFactory.to_dict(rule)
        
        return {
            "constraints": constraints_dict
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ConstraintStore:
        """
        Deserialize ConstraintStore from JSON dict.
        
        Args:
            data: Dict with 'constraints' key
        
        Returns:
            ConstraintStore instance
        """
        store = ConstraintStore()
        
        constraints_dict = data.get("constraints", {})
        for pref_key, rule_data in constraints_dict.items():
            try:
                rule = ConstraintRuleFactory.from_dict(rule_data)
                if rule is not None:
                    # Parse pref_key: "s{slice}_p{peak}_{name}"
                    parts = pref_key.split('_', 2)
                    if len(parts) >= 3 and parts[0].startswith('s') and parts[1].startswith('p'):
                        slice_id = int(parts[0][1:])
                        peak_id = int(parts[1][1:])
                        name = parts[2]
                        
                        # Import ParamRef lazily to avoid circular import
                        from nmrFit_v0 import ParamRef
                        pref = ParamRef(slice_id=slice_id, peak_id=peak_id, name=name)
                        store.add_constraint(pref, rule)
            except Exception:
                # Skip malformed constraint data
                continue
        
        return store