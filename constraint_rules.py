from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Tuple, Iterable, Set
from enum import Enum
import numpy as np  # Add missing import for np.isfinite

# Use TYPE_CHECKING to avoid circular import at runtime
# nmrFit_v0 → constraint_rules, constraint_rules → nmrFit_v0 creates cycle
if TYPE_CHECKING:
    from nmrFit_v0 import Peak, LinkExpr

@dataclass(frozen=True)
class ParamRef:
    slice_id: int
    peak_id: int
    name: str             # "pos" | "amp" | "lor" | "gauss"

def fmt_pref(pref: ParamRef) -> str:
    return f"s{pref.slice_id}_p{pref.peak_id}_{pref.name.lower()}"

def txt_to_pref(txt: str) -> ParamRef:
    """
    Convert text like "s15_p1_amp" into ParamRef.
    Raises ValueError for malformed text.
    """
    if txt is None:
        return None
    
    t = txt.strip().lower()
    if txt is None:
        return None

    # tolerate GUI placeholders / empty edits
    if isinstance(txt, str):
        t = txt.strip().lower()
    else:
        raise ValueError(f"ParamRef '{txt}' must be a string")

    if t == "" or t == "(none)":
        return None

    name_map = {
        "pos": "pos", "position": "pos", "freq": "pos", "pos_hz": "pos", "pos_ppm": "pos",
        "amp": "amp", "area": "amp", "amplitude": "amp",
        "lor": "lor", "lorentz": "lor", "lor_hz": "lor", "lorentz_hz": "lor",
        "gauss": "gauss", "gauss_disp": "gauss",
    }
    try:
        s_part, p_part, name_part = t.split("_", 2)
        slice_id = int(s_part[1:])
        peak_id = int(p_part[1:])
        normalized_name = name_part.strip().lower()

        if normalized_name not in name_map:
            raise ValueError(f"ParamRef '{txt}' has invalid parameter name '{name_part}'")
        name = name_map.get(normalized_name)
        return ParamRef(slice_id=slice_id, peak_id=peak_id, name=name)
    
    except Exception as e:
        raise ValueError(f"Malformed ParamRef '{txt}': {e}")

def pref_equal(a: ParamRef, b: ParamRef) -> bool:
        return int(a.slice_id) == int(b.slice_id) and int(a.peak_id) == int(b.peak_id) and str(a.name).lower() == str(b.name).lower()

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _get_peak_value(peak: 'Peak', param_name: str) -> float:
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

class ConstraintType(Enum):
    """Enumeration of supported constraint types."""
    LINEAR = "LINEAR"
    RELAX_EXP = "RELAX_EXP"

# Registry for mapping ConstraintType to ConstraintRule subclass
CONSTRAINT_RULE_REGISTRY: Dict[ConstraintType, Any] = {}

def register_constraint_rule(ct: ConstraintType, cls: Any) -> None:
    """
    Register a rule class for a ConstraintType. See at the bottom of this file.
    Use explicitly (static dict style) so extension is tracked.
    """
    CONSTRAINT_RULE_REGISTRY[ct] = cls



def _normalize_txt(type_txt: str) -> str:
    return (type_txt or "").strip().upper()

def resolve_constraint_type(type_txt: str) -> "ConstraintType":
    """
    Convert UI Type cell text into a ConstraintType.
    Raises ValueError for unknown types.
    """
    t = _normalize_txt(type_txt).strip().upper()
    if t in ("LINEAR",):
        return ConstraintType.LINEAR
    if t in ("RELAX_EXP", "RELAX", "DECAY", "EXP_DECAY"):
        return ConstraintType.RELAX_EXP
    raise ValueError(f"Unknown constraint type '{type_txt}'")

def row_to_rule(
    *,
    target_pref: "ParamRef",
    type_txt: str,
    driver_txt: str,
    expr_txt: str,
    enabled: bool = True,
) -> Optional["ConstraintRule"]:
    """
    Convert a LinkManagerDialog row text into a concrete ConstraintRule.

    This function:
      - normalizes & resolves ConstraintType
      - call appropriate create_rule of a ConstraintRule subclass
      - returns a ConstraintRule instance or None (meaning 'remove constraint').

    """
    if target_pref is None:
        raise ValueError("row_to_rule requires target_pref")

    expr = (expr_txt or "").strip()
    if expr == "":
        return None  # interpret empty Expr as "no constraint"

    ct = resolve_constraint_type(type_txt)

    rule_cls = CONSTRAINT_RULE_REGISTRY.get(ct)
    if rule_cls is None:
        raise ValueError(f"No class registered for {ct!r}")

    create_rule = getattr(rule_cls, "create_rule", None)
    if callable(create_rule):
        # Parse driver_pref (may be None/empty)
        driver_pref = None
        if driver_txt and str(driver_txt).strip():
            try:
                driver_pref = txt_to_pref(driver_txt)
            except Exception:
                driver_pref = None
        
        rule = create_rule(
            target_pref=target_pref,
            driver_pref=driver_pref,
            expr_txt=expr,
            enabled=enabled,
        )
        return rule
    else:
        raise ValueError(f"ConstraintRule class {rule_cls.__name__} has no create_rule method")

@dataclass
class ConstraintValidationError:
    """Container for constraint validation errors."""
    target_pref: Any  # ParamRef (deferred to avoid circular import)
    message: str


class ConstraintRule(ABC):
    """
    Abstract base class for all constraint rules.
    
    A constraint rule defines a mathematical relationship between a target parameter
    and optionally a driver parameter. Rules can be validated and applied to lmfit
    Parameter objects.
    """
    
    enabled: bool = True
    
    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType:
        """Return the type of this constraint."""
        raise NotImplementedError

    def rule_to_expr(self) -> str:
        """Optional: return a canonical RHS expression suitable for the UI."""
        return ""

    @classmethod
    @abstractmethod
    def create_rule(
        cls,
        *,
        target_pref: "ParamRef",
        driver_pref: Optional["ParamRef"],
        expr_txt: str,
        enabled: bool = True,
    ) -> "ConstraintRule":
        """Create a rule instance from the LinkManagerDialog row cells."""
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    @abstractmethod
    def apply_to_lmfit(
        self,
        params,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        slice_states: Dict[int, Any]
    ) -> None:
        """
        Apply constraint to lmfit Parameters.
        
        For same-slice constraints: set Parameter.expr (lmfit evaluates)
        For cross-slice constraints: compute value and freeze (vary=False)
        
        Args:
            params: lmfit.Parameters instance
            target_pref: ParamRef for target parameter
            driver_peak: Peak object for driver (needed for cross-slice computation)
            slice_states: Dict mapping slice_id → SliceFitState
        """
        raise NotImplementedError


class LinearConstraint(ConstraintRule):
    """
    Linear constraint: target = a * driver + b
    
    A driver parameter is required. For numeric constraints, use PeakTable fix flags instead.
    """
    
    def __init__(
        self,
        *,
        target_pref: ParamRef,
        driver_pref: ParamRef,
        a: float = 1.0,
        b: float = 0.0,
        enabled: bool = True,
        expr_txt: str = "",
    ):
        """
        Initialize a linear constraint.
        
        Args:
            target_pref: Parameter being constrained
            driver_pref: Parameter driving the constraint (REQUIRED)
            a: Multiplicative coefficient
            b: Additive offset
            expr_txt: Expression text (for display/serialization)
            enabled: Whether this constraint is active
        """
        self.target_pref = target_pref
        self.driver_pref = driver_pref
        self.a = float(a)
        self.b = float(b)
        self.enabled = bool(enabled)
        self.expr_txt = str(expr_txt or "")
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.LINEAR
#
    #@property
    #def expr(self) -> str:
    #    """Canonical RHS expression for UI display."""
    #    return self.rule_to_expr()
#   
    @staticmethod
    def parse_numeric_mult(txt: str) -> Tuple[float, str]:
        """Return (a, driver_txt) from strings like '-s1_p0_amp', '2*s1_p0_amp', '2*0.5*s1_p0_amp'."""
        t = (txt or "").strip().replace(" ", "")
        if not t:
            raise ValueError("Empty driver expression")

        sign = 1.0
        if t.startswith("+"):
            t = t[1:]
        elif t.startswith("-"):
            sign = -1.0
            t = t[1:]

        if "*" not in t:
            if _is_float(t):
                raise ValueError("Numeric-only RHS is a fixed value; do not treat it as a driver")
            if t and t[0].isdigit():
                raise ValueError("Expected '*' between numeric factor and driver, e.g. '2*s0_p0_amp'")
            return sign * 1.0, t

        factors = [f for f in t.split("*") if f]
        number = sign * 1.0
        driver_txt: Optional[str] = None
        for f in factors:
            if _is_float(f):
                number *= float(f)
            else:
                if driver_txt is not None:
                    raise ValueError(f"Multiple driver tokens found in '{txt}'")
                driver_txt = f
        if not driver_txt:
            raise ValueError(f"No driver token found in '{txt}'")
        return float(number), driver_txt

    @staticmethod
    def parse_rhs(UI_rhs_raw: str) -> Tuple[float, float, str]:  # driver_txt no longer Optional
        """
        rhs = right-hand side
        Parse forms like:
            "s15_p1_amp<br>"
            "-s15_p1_amp<br>"
            "2*s15_p1_amp<br>"
            "s15_p1_amp + 1<br>"
            "2*s15_p1_amp - 0.5<br>"
        Do not support:
            "1" (numeric only - use PeakTable fix flags)
            "2s15_p1_amp + 1"  (missing * between numberic factor and paramRef)
        """
        rhs = (UI_rhs_raw or "").strip()
        if _is_float(rhs):
            raise ValueError("Numeric-only constraint not allowed. Use PeakTable fix flags for fixed values")

        rhs = rhs.replace(" ", "")
        split_i = -1
        for i, ch in enumerate(rhs[1:], start=1): #start = 1 to match with string index of rhs
            if ch in "+-":
                split_i = i
                break

        b = 0.0
        a_driver = rhs
        if split_i != -1:
            left, right = rhs[:split_i], rhs[split_i:]
            # Prefer parsing numeric intercept on either side.
            if _is_float(right):
                b = float(right)
                a_driver = left
            elif _is_float(left):
                b = float(left)
                a_driver = right
            else:
                # assume intercept absent; treat entire RHS as driver term
                b = 0.0
                a_driver = rhs

        a, driver_txt = LinearConstraint.parse_numeric_mult(a_driver)
        if not driver_txt:
            raise ValueError("A driver is required. Numeric constraint should be done using the PeakTable and fix flags")
        
        return float(a), float(b), driver_txt  # driver_txt guaranteed non-None
    
    @staticmethod
    def parse_kv(s: str) -> dict:
        out = {}
        for chunk in s.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Expected key=value in '{chunk}'")
            k, v = chunk.split("=", 1)
            k = k.strip()
            v = v.strip()
            # don't force to float yet; some can be symbolic (T_name)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        return out
    
    @staticmethod
    def parse_UItext(UI_txt: str) -> Tuple[float, float, str]:  # driver_txt no longer Optional
        """
        Parse forms like:
            those supported by parse_rhs(), plus:
            "s14_p2_amp = 2*s15_p1_amp + 1"
            "a=0.95, b=0.02, driver=s15_p1_amp<br>"
        """
        t = (UI_txt or "").strip()
        if not t:
            raise ValueError("Empty expression")

        # KV style: 'a=..., b=..., driver=...'
        if ("driver=" in t) or ("a=" in t) or ("b=" in t):
            kv = LinearConstraint.parse_kv(t)
            a = float(kv.get("a", 1.0))
            b = float(kv.get("b", 0.0))
            driver_txt = kv.get("driver", None)
            if not driver_txt or str(driver_txt).strip() == "":
                raise ValueError("A driver is required. Numeric constraint should be done using the PeakTable and fix flags")
            driver_txt = str(driver_txt).strip()
            return a, b, driver_txt

        # Full assignment: lhs = rhs
        if "=" in t and not t.startswith("="):
            _lhs, rhs = t.split("=", 1)
            return LinearConstraint.parse_rhs(rhs.strip())

        # RHS only
        return LinearConstraint.parse_rhs(t)
    
    def UItext_to_expr(UItext) -> str:
        """Convert UI expression text into standard RHS expression for UI display and lmfit."""
        a, b, drv = LinearConstraint.parse_UItext(UItext)
        # drv is guaranteed to be non-empty string at this point

        if a == 1.0 and b == 0.0:
            return drv
        if a == 1.0:
            return f"{drv}+{b}" if b >= 0 else f"{drv}-{abs(b)}"
        if b == 0.0:
            return f"{a}*{drv}"
        return f"{a}*{drv}+{b}" if b >= 0 else f"{a}*{drv}-{abs(b)}"
    
    def rule_to_expr(self) -> str:
        """Safely call expr_txt or construct expr from a b driver."""
        # Prefer canonical stored text, but canonicalize defensively.
        
        s = (self.expr_txt or "").strip()
        if s:
            return self.expr_txt

        drv = fmt_pref(self.driver_pref)
        a = float(self.a)
        b = float(self.b)

        if a == 1.0 and b == 0.0:
            return drv
        if a == 1.0:
            return f"{drv}+{b}" if b >= 0 else f"{drv}-{abs(b)}"
        if b == 0.0:
            return f"{a}*{drv}"
        return f"{a}*{drv}+{b}" if b >= 0 else f"{a}*{drv}-{abs(b)}"

    @classmethod
    def create_rule(
        cls,
        *,
        target_pref: ParamRef,
        driver_pref: Optional[ParamRef],
        expr_txt: str,  # Changed from UI_txt to match signature
        enabled: bool = True,
    ) -> "ConstraintRule":
        """Create a LinearConstraint from UI driver + expr cells."""
        a, b, driver_txt = cls.parse_UItext(expr_txt)
        canonical_expr = cls.UItext_to_expr(expr_txt)
        
        # Parse driver from expression
        drv_from_expr = txt_to_pref(driver_txt)  # driver_txt is guaranteed non-None

        # Validate driver consistency
        if driver_pref is not None:
            if not pref_equal(driver_pref, drv_from_expr):
                raise ValueError(
                    f"Driver mismatch: driver cell '{fmt_pref(driver_pref)}' "
                    f"!= expr driver '{fmt_pref(drv_from_expr)}'"
                )
            final_driver = driver_pref
        else:
            final_driver = drv_from_expr

        return cls(
            target_pref=target_pref,
            driver_pref=final_driver,
            a=a,
            b=b,
            enabled=enabled,
            expr_txt=canonical_expr
        )
    
    def validate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> List[str]:
        """Validate semantic preconditions: driver exists and coefficients are finite."""
        errors = []
        
        # Driver must exist for LinearConstraint (semantic check)
        if driver_peak is None:
            errors.append(
                f"Linear constraint on {fmt_pref(target_pref)}: "
                f"driver peak s{self.driver_pref.slice_id}_p{self.driver_pref.peak_id} not found"
            )
        
        # Validate coefficients
        if not np.isfinite(self.a):
            errors.append(
                f"Linear constraint on {fmt_pref(target_pref)}: "
                f"coefficient a={self.a} is not finite"
            )
        
        if not np.isfinite(self.b):
            errors.append(
                f"Linear constraint on {fmt_pref(target_pref)}: "
                f"offset b={self.b} is not finite"
            )
        
        return errors
    
    def apply_to_lmfit(
        self,
        params,
        target_pref: ParamRef,
        driver_peak: Peak,
        slice_states: Dict[int, Any],
        enabled: bool = True,
        vary: bool = True
    ) -> None:
        """
        Apply linear constraint to lmfit.
        
        Same-slice: set Parameter.expr = "a*driver_key + b" and Vary=True
        Cross-slice: Sequential -> compute value and freeze. Joint -> use expr and Vary=True
        """
        if enabled and vary == True:
            param_key = fmt_pref(target_pref)

            if param_key not in params:
                return

            param = params[param_key]
            driver_key = fmt_pref(self.driver_pref)

            # Check if driver is in same slice (can use algebraic expr)
            if int(self.driver_pref.slice_id) == int(target_pref.slice_id):
                # Same-slice: use Parameter.expr
                expr_lmfit = self.rule_to_expr()
                param.set(expr=expr_lmfit)

        else:
            # Cross-slice: compute value and freeze
            if driver_peak is None:
                return  # validation should have caught this
            driver_value = _get_peak_value(driver_peak, self.driver_pref.name)
            computed_value = float(self.a * driver_value + self.b)
            param.set(value=computed_value, expr=expr_lmfit, vary=False)

class RelaxDecayConstraint(ConstraintRule):
    """
    Relaxation exponential constraint: target = driver * A * exp(-t / T) + C
    
    Used for fitting relaxation curves where target values depend on acquisition time.
    Time can be supplied explicitly or read from slice_states.
    """
    
    def __init__(
        self,
        target_pref: ParamRef,
        driver_pref: ParamRef,
        T_seconds: float = None,
        A: float = 1.0,
        C: float = 0.0,
        time_seconds: Optional[float] = None,
        T_name: Optional[str] = None,
        expr_txt: str = "",
        enabled: bool = True
    ):
        """
        Initialize a relaxation exponential constraint.
        
        Args:
            target_pref: Parameter being constrained
            driver_pref: Parameter driving the decay
            T_seconds: Decay time constant (seconds) can be None if T_name used
            A: Amplitude multiplicative factor
            C: Baseline offset
            time_seconds: Explicit time value (overrides T_name lookup)
            T_name: Name of time constant can be None if T_seconds used
            enabled: Whether this constraint is active
        """
        self.target_pref = target_pref
        self.driver_pref = driver_pref
        self.T_seconds = float(T_seconds) if T_seconds is not None else None
        self.A = float(A)
        self.C = float(C)
        self.time_seconds = float(time_seconds) if time_seconds is not None else None
        self.T_name = T_name
        self.enabled = bool(enabled)
        self.expr_txt = str(expr_txt or "")
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.RELAX_EXP
    
    @classmethod
    def parse_rhs(UI_rhs_raw: str) -> Dict: #
        """
            "s15_p1_amp*1*exp(-0.1/T_name + 2)<br>"
            return dict with keys of A, driver_txt, C, t_override, T_name, T_number
        """
        out = {}
        txt = UI_rhs_raw.strip().lower()   
        exp_idx = txt.find("exp(")
        if exp_idx == -1:
            raise ValueError(f"Malformed exponential constraint expression: '{UI_rhs_raw}'")

        left = txt[:exp_idx] # Left of *exp
        right = txt[exp_idx+1:]  # Right of exp(...) to end. ignore the '*'

        # parse left side to get A and driver_txt
        A, driver_txt = LinearConstraint.parse_numeric_mult(left.replace("*", "").strip())
        
        # parse right side to get t, T_name or T, C
        s = right.replace("exp(", "").replace(")", "")
        C = 0.0
        idx_plus = s.find("+", 1)
        idx_minus = s.find("-", 1)
        indices = [i for i in (idx_plus, idx_minus) if i != -1]
        i = min(indices) if indices else -1
        if i != -1:
            main = s[:i].replace(" ", "")
            c_part = s[i:].replace(" ", "")
            C = float(c_part)
        else:
            main = s
        # main should look like: -0.1/T_name   or  -0.1/0.035
        if not main.startswith("-"):
            raise ValueError(f"Expected '-' at start of exponential argument in '{UI_rhs_raw}'")
        if main.startswith("-"):
            main_k = main
        if "/" not in main_k:
            raise ValueError(f"Expected form like '-k/T_name' in '{UI_rhs_raw}'")
        k_part, T_part = main_k.split("/", 1)
        k_val = float(k_part)
        # T_part can be numeric or name
        if _is_float(T_part):
            # numeric T
            # original exp() used "T"
            out["T"] = float(T_part)
        else:
            out["T_name"] = T_part

        out = {"A": A, "driver_txt": driver_txt, "C": C, "t_override": k_val, "T_name": T_part, "T": float(T_part) if _is_float(T_part) else None}

        return out
    
    @classmethod
    def parse_UItext(UI_txt) -> Dict: #keys are A, driver_txt, C, t_override, T_name, T_number
        """Parse the expression string into components.
            similar to parse_rhs plus
            "exp(A=1.2, T=0.035, C=0)<br>"
            "exp(A=1.2, T_name=Tglobal, C=0)<br>"
            "s14_p2_amp = s15_p1_amp*1*[exp(-0.1/T_name) + 2]<br>"
            "s14_p2_amp = exp(A=1.2, T_name=Tglobal, C=0)
            return dict with keys of A, driver_txt, C, t_override, T_name, T_number
        """
        out = {}
        t = UI_txt.strip()
                # Case 1: pure kv-style exp(...)
                        # 1) KV style: "exp(A=1.2, T=0.035, C=0)" --> return dict with A,T_seconds,T_name,C
                # 1) rhs, e.g "2*s15_p1_amp + 1"
        if "=" not in t: #no assignment, just rhs
            out = RelaxDecayConstraint.parse_rhs(t)
            return out

        # 2) full assignment: lhs = rhs, e.g "s14_p2_amp = 2*s15_p1_amp + 1"

        if "=" in t and not t.startswith("="): #detect full assignment
            lhs, rhs = t.split("=", 1)
            rhs = rhs.strip() #2*s15_p1_amp + 1
            out = RelaxDecayConstraint.parse_rhs(rhs)
            return out

        # 3) dict-like assignment: A=1, C=2, driver=s14_p2_amp

        if "driver=" in t or "A=" in t or "C=" in t:
            kv = LinearConstraint.parse_kv(t)
            A = float(kv.pop("A", 1.0))
            C = float(kv.pop("C", 0.0))
            driver_txt = str(kv.pop("driver", ""))
            t_override = str(kv.pop("t", ""))
            T_name = str(kv.pop("T_name", None))
            T_number = kv.pop("T", None)
            out = {"A": A, "driver_txt": driver_txt, "C": C, "t_override": t_override, "T_name": T_name, "T_number": float(T_number) if T_number is not None else None}
            return out


    @classmethod
    def create_rule(
        cls,
        *,
        target_pref: ParamRef,
        driver_pref: Optional[ParamRef],
        expr_txt: str,
        enabled: bool = True,
    ) -> "ConstraintRule":
        
        """Create a RelaxDecayConstraint from UI driver + expr cells."""
        out = cls.parse_UItext(expr_txt)
        A = float(out.get("A", 1.0))
        driver_txt = out.get("driver_txt", None)
        C = out.get("C", 0.0)
        t_override = out.get("t_override", None)
        T_name = out.get("T_name", None)
        T_number = out.get("T_number", None)

        drv_from_expr: Optional[ParamRef] = None
        if driver_txt:
            drv_from_expr = txt_to_pref(str(driver_txt).strip())

        if driver_pref is not None and drv_from_expr is not None:
            if not pref_equal(driver_pref, drv_from_expr):
                raise ValueError(
                    f"Driver mismatch: driver cell '{cls._fmt_pref(driver_pref)}' "
                    f"!= expr driver '{cls._fmt_pref(drv_from_expr)}'"
                )
        if driver_pref is None:
            driver_pref = drv_from_expr
        if driver_pref is None:
            raise ValueError("RELAX_EXP requires a driver parameter")

        # time override is required for evaluation; store as float if possible
        t_s: Optional[float] = None
        if t_override not in (None, ""):
            t_s = float(t_override)

        # Build a canonical expr for display (and optional debugging)
        canonical = ""
        drv_txt = cls._fmt_pref(driver_pref)
        if T_number is not None:
            expr_txt = f"{A}*{driver_txt}*exp(-{t_override}/{T_number}) + {C}"
            if A == 1 and C == 0:
                expr_txt = f"{driver_txt}*exp(-{t_override}/{T_number})"
        elif T_name is not None:
            expr_txt = f"{A}*{driver_txt}*exp(-{t_override}/{T_name}) + {C}"
            if A == 1 and C == 0:
                expr_txt = f"{driver_txt}*exp(-{t_override}/{T_name})"

        rule = RelaxDecayConstraint(
            target_pref=target_pref,
            driver_pref=driver_pref if driver_pref is not None else None,
            T_seconds=T_number,
            A=A,
            C=C,
            time_seconds=t_override,
            T_name=T_name,
            enabled=enabled
        )
        return rule
    
    def validate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: Optional[ParamRef],
        slice_states: Dict[int, Any]
    ) -> List[str]:
        """Validate exponential constraint semantic preconditions."""
        errors = []
        
        # Driver must exist
        if driver_peak is None or driver_pref is None:
            errors.append(
                f"Exponential constraint on {fmt_pref(target_pref)}: "
                f"driver peak s{self.driver_pref.slice_id}_p{self.driver_pref.peak_id} not found"
            )
        
        # Validate T (if numeric) must be positive
        if self.T_seconds is not None:
            try:
                T_val = float(self.T_seconds)
                if T_val <= 0.0:
                    errors.append(
                        f"Exponential constraint on {fmt_pref(target_pref)}: "
                        f"time constant T={T_val} must be positive"
                    )
            except (ValueError, TypeError):
                errors.append(
                    f"Exponential constraint on {fmt_pref(target_pref)}: "
                    f"time constant T={self.T_seconds} is not a valid number"
                )
        
        # Check time availability (either explicit or from slice_states)
        if self.time_seconds is None and self.T_name is None:
            # Try to get from slice_states as fallback
            slice_id = int(target_pref.slice_id)
            state = slice_states.get(slice_id)
            t_available = False
            if state is not None:
                t_f1 = getattr(state, 't_f1', None)
                if t_f1 is not None and hasattr(t_f1, '__getitem__'):
                    try:
                        _ = float(t_f1[slice_id])
                        t_available = True
                    except (IndexError, TypeError, ValueError):
                        pass
            
            if not t_available:
                errors.append(
                    f"Exponential constraint on {fmt_pref(target_pref)}: "
                    f"no time value available (no time_seconds or T_name, and slice_states has no t_f1)"
                )
        
        if self.time_seconds is not None and self.time_seconds < 0.0:
            errors.append(
                f"Exponential constraint on {fmt_pref(target_pref)}: "
                f"time_seconds={self.time_seconds} must be non-negative"
            )
        
        # Validate coefficients
        if not np.isfinite(self.A):
            errors.append(
                f"Exponential constraint on {fmt_pref(target_pref)}: "
                f"amplitude A={self.A} is not finite"
            )
        
        if not np.isfinite(self.C):
            errors.append(
                f"Exponential constraint on {fmt_pref(target_pref)}: "
                f"baseline C={self.C} is not finite"
            )
        
        return errors
    
    def apply_to_lmfit(
        self,
        params,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        slice_states: Dict[int, Any]
    ) -> None:
        """
        Apply exponential constraint to lmfit.
        
        For exponential constraints, set the parameter to fixed value
        (expression building would be complex; evaluation is done once upfront).
        """
        param_key = fmt_pref(target_pref)
        
        if param_key not in params:
            return
        
        if driver_peak is None:
            return  # validation should have caught this
        
        # Get time value
        t = self._get_time_value(target_pref, slice_states)
        
        # Get T value (numeric or from registry)
        T = self._get_T_value(slice_states)
        
        # Extract driver parameter value
        driver_value = _get_peak_value(driver_peak, self.driver_pref.name)
        
        # Compute: driver * A * exp(-t / T) + C
        exp_term = np.exp(-t / T)
        computed_value = float(driver_value * self.A * exp_term + self.C)
        
        # Freeze parameter
        param = params[param_key]
        param.value = computed_value
        param.vary = False
    
    def _get_time_value(self, target_pref: ParamRef, slice_states: Dict[int, Any]) -> float:
        """Get time value from explicit value or from slice_states."""
        if self.time_seconds is not None:
            return float(self.time_seconds)
        
        # Try to get from slice_states t_f1 array
        slice_id = int(target_pref.slice_id)
        state = slice_states.get(slice_id)
        if state is not None:
            t_f1 = getattr(state, 't_f1', None)
            if t_f1 is not None and hasattr(t_f1, '__getitem__'):
                try:
                    return float(t_f1[slice_id])
                except (IndexError, TypeError):
                    pass
        
        raise ValueError(f"No time value available for slice {slice_id}")
    
    def _get_T_value(self, slice_states: Dict[int, Any]) -> float:
        """Get T value from numeric T_seconds or from T_name registry."""
        if self.T_seconds is not None:
            return float(self.T_seconds)
        
        if self.T_name is not None:
            # Lookup from MainWindow._TSeedRegistry (would need to pass in)
            # For now, raise to indicate missing implementation
            raise NotImplementedError(
                f"T_name lookup for '{self.T_name}' requires registry access; "
                f"use explicit T_seconds instead"
            )
        
        raise ValueError("No T value available for exponential constraint")


class ConstraintRuleFactory:
    """
    Factory for creating constraint rules from serialized representations.
    Useful for loading/saving constraints to file and creating rules from user input.
    """
    
    @staticmethod
    def parse_row_to_constraint(*, target, type_txt: str, driver, expr_txt: str):
        """
        Single place to convert a 4-cell row into ConstraintRule.
        Try in this order:
        1) if expr_txt starts with '=' → inline
        2) else if type is RELAX_EXP → parse KV style
        3) else → linear implicit
        """
        if expr_txt.startswith("="):
            return ConstraintRuleFactory.parse_UItext(target, expr_txt.lstrip("="))

        if type_txt.upper() == "RELAX_EXP":
            return ConstraintRuleFactory.parse_inline_or_kv(target, expr_txt)

        if driver is not None and not expr_txt:
            return LinearConstraint(
                target_pref=target,
                driver_pref=driver,
                a=1.0,
                b=0.0,
                enabled=True,
            )

        return ConstraintRuleFactory.parse_UItext(target, expr_txt)


    @staticmethod
    def create_linear_constraint(target_pref, driver_pref, a=1.0, b=0.0, enabled=True):
        """
        Create a LinearConstraint directly from user input parameters.
        """
        return LinearConstraint(
            target_pref=target_pref,
            driver_pref=driver_pref,
            a=a,
            b=b,
            enabled=enabled
        )

    @staticmethod
    def create_relax_decay_constraint(
        target_pref,
        driver_pref,
        T_seconds,
        A=1.0,
        C=0.0,
        time_seconds=None,
        T_name=None,
        enabled=True
    ):
        """
        Create a RelaxDecayConstraint directly from user input parameters.
        """
        return RelaxDecayConstraint(
            target_pref=target_pref,
            driver_pref=driver_pref,
            T_seconds=T_seconds,
            A=A,
            C=C,
            time_seconds=time_seconds,
            T_name=T_name,
            enabled=enabled
        )


# ============================================================================
# Migration Bridge: LinkExpr → ConstraintRule
# ============================================================================

def LinkExpr_to_ConstraintRule(link_expr) -> ConstraintRule:
    """
    Convert old LinkExpr to new ConstraintRule subclass.
    
    This function enables backward compatibility when loading existing fit files
    that use the deprecated LinkExpr format. It infers the type from LinkExpr.type
    property and creates appropriate ConstraintRule.
    
    Args:
        link_expr: LinkExpr instance with target, driver, args, enabled
        
    Returns:
        ConstraintRule subclass instance (LinearConstraint or RelaxDecayConstraint)
        
    Raises:
        ValueError: if LinkExpr format is invalid
    """
    link_type = link_expr.type  # Use property to infer type from args
    
    if link_type == "LINEAR":
        return LinearConstraint(
            target_pref=link_expr.target,
            driver_pref=link_expr.driver,
            a=link_expr.args.get("a", 1.0),
            b=link_expr.args.get("b", 0.0),
            enabled=link_expr.enabled
        )
    
    elif link_type == "RELAX_EXP":
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
        raise ValueError(f"Unknown link type inferred from LinkExpr: {link_type}")


def ConstraintRule_to_LinkExpr(rule: ConstraintRule):
    """
    Convert new ConstraintRule to old LinkExpr format.
    
    This function enables backward compatibility when exporting to the old
    LinkExpr format. It is the inverse of LinkExpr_to_ConstraintRule.
    
    Args:
        rule: ConstraintRule subclass instance
        
    Returns:
        LinkExpr instance with target, driver, args, enabled
        
    Raises:
        ValueError: if rule type is unknown
    """
    # Import at function level to avoid circular dependency
    from nmrFit_v0 import LinkExpr as _LinkExpr
    
    if isinstance(rule, LinearConstraint):
        return _LinkExpr(
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
        Retrieve a constraint rule for the given target parameter reference.
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
    Per-slice registry of constraints.
    Maps ParamRef (target) → ConstraintRule.
    """
    _constraints: Dict[ParamRef, ConstraintRule] = field(default_factory=dict) #dict {target, rule}
    _dependents: Dict[ParamRef, Set[ParamRef]] = field(default_factory=dict) #{driver, set(targets)}
    
    def add_constraint(self, rule) -> None:
        """Add or update a constraint"""
        # Remove old reverse dependency
        old = self._constraints.get(rule.target_pref)
        if old and hasattr(old, 'driver_pref') and old.driver_pref:
            self._dependents.get(old.driver_pref, set()).discard(rule.target_pref)

        self._constraints[rule.target_pref] = rule
        if rule.driver_pref:
            self._dependents.setdefault(rule.driver_pref, set()).add(rule.target_pref)

    def remove_constraint(self, target: ParamRef) -> None:
        old = self._constraints.pop(target, None)
        if old and old.driver_pref:
            self._dependents.get(old.driver_pref, set()).discard(target)

    def get_constraint(self, target: ParamRef) -> Optional[ConstraintRule]:
        """Get constraint for a target parameter."""
        return self._constraints.get(target)
    
    def is_constrained(self, target: ParamRef) -> bool:
        s = self._constraints.get(target)
        return bool(s and s.enabled)

    def all_constraints(self) -> Iterable[Tuple[ParamRef, ConstraintRule]]:
        """Iterate all (target, rule) pairs."""
        return self._constraints.items()
    
    def get_dependents(self, driver: ParamRef) -> List[ParamRef]:
        """Get all targets driven by this parameter."""
        return list(self._dependents.get(driver, set()))


@dataclass
class FitOrchestrator:
    """
    Orchestrates constraint validation and application for fitting.
    
    Responsibilities:
    - Validate all constraints before fitting
    - Apply constraints to lmfit.Parameters in correct order
    - Handle cross-slice dependencies
    
    Does NOT:
    - Execute fits (that's FitContext's job)
    - Parse UI text (that's row_to_rule's job)
    - Branch on constraint types (that's ConstraintRule.apply_to_lmfit's job)
    """
    constraint_store: ConstraintStore
    
    def validate_constraints(
        self,
        peak_map: Dict[Tuple[int, int], Any],  # (slice_id, peak_id) → Peak
        slice_states: Dict[int, Any],           # slice_id → SliceFitState
    ) -> List[ConstraintValidationError]:
        """
        Validate all enabled constraints.
        
        Args:
            peak_map: Mapping from (slice_id, peak_id) to Peak objects
            slice_states: Mapping from slice_id to SliceFitState
        
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        for target_pref, rule in self.constraint_store.all_constraints():
            if not getattr(rule, 'enabled', True):
                continue
            
            # Get target peak
            target_key = (int(target_pref.slice_id), int(target_pref.peak_id))
            target_peak = peak_map.get(target_key)
            
            # Get driver peak (if rule has a driver)
            driver_pref = getattr(rule, 'driver_pref', None)
            driver_peak = None
            if driver_pref is not None:
                driver_key = (int(driver_pref.slice_id), int(driver_pref.peak_id))
                driver_peak = peak_map.get(driver_key)
            
            # Validate this rule
            try:
                rule_errors = rule.validate(
                    target_peak=target_peak,
                    target_pref=target_pref,
                    driver_peak=driver_peak,
                    driver_pref=driver_pref,
                    slice_states=slice_states
                )
                
                for msg in rule_errors:
                    errors.append(ConstraintValidationError(
                        target_pref=target_pref,
                        message=msg
                    ))
            except Exception as e:
                errors.append(ConstraintValidationError(
                    target_pref=target_pref,
                    message=f"Validation exception: {e}"
                ))
        
        return errors
    
    def apply_constraints_to_lmfit(
        self,
        params,                                 # lmfit.Parameters
        peak_map: Dict[Tuple[int, int], Any],  # (slice_id, peak_id) → Peak
        slice_states: Dict[int, Any],           # slice_id → SliceFitState
    ) -> None:
        """
        Apply all enabled constraints to lmfit.Parameters.
        
        Each ConstraintRule owns its own apply_to_lmfit() logic.
        This method just coordinates the sequencing.
        
        Args:
            params: lmfit.Parameters instance to configure
            peak_map: Mapping from (slice_id, peak_id) to Peak objects
            slice_states: Mapping from slice_id to SliceFitState
        """
        for target_pref, rule in self.constraint_store.all_constraints():
            if not getattr(rule, 'enabled', True):
                continue
            
            # Get driver peak (if rule has a driver)
            driver_pref = getattr(rule, 'driver_pref', None)
            driver_peak = None
            if driver_pref is not None:
                driver_key = (int(driver_pref.slice_id), int(driver_pref.peak_id))
                driver_peak = peak_map.get(driver_key)
            
            # Delegate to rule's apply logic
            try:
                rule.apply_to_lmfit(
                    params=params,
                    target_pref=target_pref,
                    driver_peak=driver_peak,
                    slice_states=slice_states
                )
            except Exception as e:
                # Log but continue (defensive; validation should have caught this)
                import logging
                logging.getLogger("fit").warning(
                    f"Failed to apply constraint for {fmt_pref(target_pref)}: {e}"
                )

register_constraint_rule(ConstraintType.LINEAR, LinearConstraint)
register_constraint_rule(ConstraintType.RELAX_EXP, RelaxDecayConstraint)