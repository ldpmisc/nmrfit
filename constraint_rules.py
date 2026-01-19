from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Tuple, Iterable, Set, Callable, Mapping
from enum import Enum
from matplotlib.units import registry
import numpy as np  # Add missing import for np.isfinite
from fit_types import ParamRef, ParamBounds

import logging

log = logging.getLogger("fit")


# Use TYPE_CHECKING to avoid circular import at runtime
# nmrFit_v0 → constraint_rules, constraint_rules → nmrFit_v0 creates cycle
if TYPE_CHECKING:
    from nmrFit_v0 import Peak, LinkExpr


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

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

# --- FIT-TIME HELPERS (module-level) --
def resolve_k_from_Tseed_registry(
    *,
    T_name: str,
    Tseed_registry: Mapping[str, Mapping[str, Any]],
    default_T_seed_s: float = 0.001,
    default_T_lo_s: Optional[float] = None,
    default_T_hi_s: Optional[float] = None,
    default_fixed: bool = False,
    eps: float = 1e-15,
) -> Tuple[float, ParamBounds, bool]:
    """
    SSOT keys expected in tseed_registry[T_name]:
      - "T_seed_s": float
      - "T_lo_s": Optional[float]
      - "T_hi_s": Optional[float]
      - "fixed": bool

    Returns (k_seed, k_bounds, fixed), where k = 1/T and bounds are mapped from T-bounds:
      T in [T_lo, T_hi] (positive) => k in [1/T_hi, 1/T_lo]
    """
    name = (T_name or "").strip()
    if not name:
        raise ValueError("Empty T_name.")

    rec = Tseed_registry.get(name)
    if rec is None:
        log.warning("T_name '%s' missing in TSeed registry; using defaults.", name)
        fixed = bool(default_fixed)
        T_seed = float(default_T_seed_s)
        T_lo = default_T_lo_s
        T_hi = default_T_hi_s
    else:
        fixed = bool(rec.get("fixed", default_fixed))
        T_seed = rec.get("T_seed_s", None)
        T_seed = float(T_seed) if T_seed is not None else float(default_T_seed_s)
        T_lo = rec.get("T_lo_s", None)
        T_hi = rec.get("T_hi_s", None)
        T_lo = float(T_lo) if T_lo is not None else default_T_lo_s
        T_hi = float(T_hi) if T_hi is not None else default_T_hi_s

    if T_seed <= 0:
        raise ValueError(f"T_seed_s must be > 0 for '{name}', got {T_seed}.")
    if (T_lo is not None) and (T_lo <= 0):
        raise ValueError(f"T_lo_s must be > 0 for '{name}', got {T_lo}.")
    if (T_hi is not None) and (T_hi <= 0):
        raise ValueError(f"T_hi_s must be > 0 for '{name}', got {T_hi}.")

    if abs(T_seed) < eps:
        raise ValueError(f"T_seed_s too small to invert safely for '{name}'.")

    k_seed = 1.0 / T_seed

    # bounds mapping
    k_lo = None  # min
    k_hi = None  # max
    if T_hi is not None:
        if abs(T_hi) < eps:
            raise ValueError(f"T_hi_s too small to invert safely for '{name}'.")
        k_lo = 1.0 / T_hi
    if T_lo is not None:
        if abs(T_lo) < eps:
            raise ValueError(f"T_lo_s too small to invert safely for '{name}'.")
        k_hi = 1.0 / T_lo

    if (k_lo is not None) and (k_hi is not None) and (k_lo > k_hi):
        raise ValueError(f"Derived k bounds invalid for '{name}': {k_lo} > {k_hi}.")

    return k_seed, ParamBounds(lo=k_lo, hi=k_hi), fixed

def ensure_k_param_in_lmfit(
    *,
    params,  # lmfit.Parameters
    T_name: str,
    Tseed_registry,
    default_kmin: float = 1e-12,
    default_kmax: float = 1e3,
) -> str:
    """
    Create or update shared parameter k__{T_name} in params using TSeed registry.
    Returns the lmfit parameter name.
    """
    k_name = f"k__{T_name}"

    k_seed, k_bounds, fixed = resolve_k_from_Tseed_registry(
        T_name=T_name,
        Tseed_registry=Tseed_registry,
    )

    kmin = default_kmin if k_bounds.lo is None else float(k_bounds.lo)
    kmax = default_kmax if k_bounds.hi is None else float(k_bounds.hi)

    if kmin <= 0:
        kmin = default_kmin
    if kmax <= kmin:
        kmax = max(default_kmax, kmin * 10)

    if k_name not in params:
        params.add(k_name, value=float(k_seed), min=float(kmin), max=float(kmax), vary=(not fixed))
    else:
        pk = params[k_name]
        # ENFORCE bounds/vary every time (SSOT)
        pk.min = float(kmin)
        pk.max = float(kmax)
        pk.vary = (not fixed)
        # Optional: if pk.value is outside bounds after UI edit, clamp
        if pk.value < pk.min:
            pk.value = pk.min
        elif pk.value > pk.max:
            pk.value = pk.max

    return k_name

class ConstraintType(Enum):
    """Enumeration of supported constraint types."""
    LINEAR = "LINEAR"
    RELAX_DECAY = "RELAX_DECAY"
    RELAX_GROWTH = "RELAX_GROWTH"

# Registry for mapping ConstraintType to ConstraintRule subclass
CONSTRAINT_RULE_REGISTRY: Dict[ConstraintType, Any] = {} #Any = ConstraintRule subclass

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
    if t in ("LINEAR", "linear"):
        return ConstraintType.LINEAR
    if t in ("RELAX_DECAY", "RELAX", "DECAY", "EXP_DECAY"):
        return ConstraintType.RELAX_DECAY
    raise ValueError(f"Unknown constraint type '{type_txt}'")

def dispatch_rule_from_ui(
    *,
    target_pref: "ParamRef",
    type_txt: str,
    driver_txt: str,
    expr_txt: str,
    enabled: bool = True,
    ctx: "ParseContext",
) -> Optional["ConstraintRule"]:
    """
    Convert a LinkManagerDialog row text into a concrete ConstraintRule.

    This function:
      - normalizes & resolves ConstraintType
      - delegates to subclass interpret_ui method
      - returns a ConstraintRule instance or None (meaning 'remove constraint').

    """
    if target_pref is None:
        raise ValueError("dispatch_rule_from_ui requires target_pref")

    expr = (expr_txt or "").strip()
    if expr == "":
        return None  # interpret empty Expr as "no constraint"

    ct = resolve_constraint_type(type_txt)

    rule_cls = CONSTRAINT_RULE_REGISTRY.get(ct)
    if rule_cls is None:
        raise ValueError(f"No class registered for {ct!r}")

    interpret_ui = getattr(rule_cls, "interpret_ui", None)
    if callable(interpret_ui):
        rule = interpret_ui(
            target_pref=target_pref,
            driver_txt=driver_txt,
            expr_txt=expr,
            enabled=enabled,
            ctx=ctx,
        )
        return rule
    else:
        raise ValueError(f"ConstraintRule class {rule_cls.__name__} has no interpret_ui method")

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
    
    @classmethod
    @abstractmethod
    def parse_UItext(cls, UI_txt: str) -> Any:
        """Parse UI text into rule-specific components."""
        pass
    
    @classmethod
    @abstractmethod
    def infer_driver_from_expr(cls, expr_txt: str) -> Optional[str]:
        """Return driver_txt if the rule can infer it; else None."""
        return None
    
    @classmethod
    @abstractmethod
    def interpret_ui(
        cls,
        *,
        target_pref: "ParamRef",
        driver_txt: str,
        expr_txt: str,
        enabled: bool,
        ctx: "ParseContext",
    ) -> Optional["ConstraintRule"]:
        """Parse UI text, resolve context (time, T seeds), perform conversions, and call create_rule."""
        pass

    @classmethod
    @abstractmethod
    def create_rule(
        cls,
        *,
        target_pref: "ParamRef",
        driver_pref: "ParamRef",
        enabled: bool = True,
        **kwargs,
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
        params: Dict[str, Any],
        registry: Dict[ParamRef, Dict[str, Any]],
        allow_external: bool = False,
        vary: bool = True,
    ) -> None:
        """
        Apply constraint to lmfit Parameters.
                
        For same-slice constraints: set Parameter.expr (lmfit evaluates)
        For cross-slice constraints: compute value and freeze (vary=False)
        
        Args:
            params: lmfit.Parameters instance
            target_pref: ParamRef for target parameter
            driver_pref: ParamRef for driver parameter -> driver_peak to get value
            slice_states: Dict mapping slice_id → SliceFitState
        """
        raise NotImplementedError

    def to_display_expr(self) -> str:
        """Optional: return a canonical RHS expression suitable for the UI."""
        return ""



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
            if is_float(t):
                raise ValueError("Numeric-only RHS is a fixed value; do not treat it as a driver")
            if t and t[0].isdigit():
                raise ValueError("Expected '*' between numeric factor and driver, e.g. '2*s0_p0_amp'")
            return sign * 1.0, t

        factors = [f for f in t.split("*") if f]
        number = sign * 1.0
        driver_txt: Optional[str] = None
        for f in factors:
            if is_float(f):
                number *= float(f)
            else:
                if driver_txt is not None:
                    raise ValueError(f"Multiple driver tokens found in '{txt}'")
                driver_txt = f
        if not driver_txt:
            raise ValueError(f"No driver token found in '{txt}'")
        return float(number), driver_txt

    @classmethod
    def parse_rhs(cls, UI_rhs_raw: str) -> Tuple[float, float, str]:  # driver_txt no longer Optional
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
        if is_float(rhs):
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
            if is_float(right):
                b = float(right)
                a_driver = left
            elif is_float(left):
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
    
    @classmethod
    def parse_UItext(cls, UI_txt: str) -> Tuple[float, float, str]:  # driver_txt no longer Optional
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
    
    @staticmethod
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
    
    def to_display_expr(self) -> str:
        """Safely call expr_txt or construct expr from a b driver."""
        
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
    
    def to_lmfit_expr(self) -> str:
        """Return lmfit-compatible expression string."""
        """ same  as to_display_expr for linear case"""
        return self.to_display_expr()

    @classmethod
    def infer_driver_from_expr(cls, expr_txt: str) -> Optional[str]:
        _a, _b, driver_txt = cls.parse_UItext(expr_txt)
        return driver_txt
    
    @classmethod
    def interpret_ui(
        cls,
        *,
        target_pref: "ParamRef",
        driver_txt: str,
        expr_txt: str,
        enabled: bool,
        ctx: "ParseContext",
    ) -> Optional["LinearConstraint"]:
        """Parse UI text and create LinearConstraint."""
        # Parse expr_txt to extract a, b, and potentially driver from expression
        a, b, expr_driver_txt = cls.parse_UItext(expr_txt)
        
        # Parse driver_pref from driver_txt
        driver_pref = None
        if driver_txt and str(driver_txt).strip():
            try:
                driver_pref = txt_to_pref(driver_txt)
            except Exception:
                driver_pref = None
        
        # If no explicit driver_txt but expr contains driver, use that
        if driver_pref is None and expr_driver_txt:
            try:
                driver_pref = txt_to_pref(expr_driver_txt)
            except Exception:
                pass
        
        if driver_pref is None:
            raise ValueError("LinearConstraint requires a driver parameter")
        
        return cls.create_rule(
            target_pref=target_pref,
            driver_pref=driver_pref,
            a=a,
            b=b,
            enabled=enabled,
            expr_txt=expr_txt,
        )

    @classmethod
    def create_rule(
        cls,
        *,
        target_pref: "ParamRef",
        driver_pref: "ParamRef",
        a: float,
        b: float,
        enabled: bool = True,
        expr_txt: str = "",
    ) -> "LinearConstraint":
        """Construct LinearConstraint from fully-resolved domain parameters."""
        return cls(
            target_pref=target_pref,
            driver_pref=driver_pref,
            a=a,
            b=b,
            enabled=enabled,
            expr_txt=expr_txt,
        )
    
    def validate(
        self,
        target_peak: Peak,
        target_pref: ParamRef,
        driver_peak: Optional[Peak],
        driver_pref: ParamRef,
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
        *,
        params: Dict[str, Any],
        registry: Dict[ParamRef, Dict[str, Any]],  # {'value': float}
        allow_external: bool = False,
        vary: bool = True,  # True=joint, False=sequential/single
    ) -> None:
        
        """
        Apply linear constraint to lmfit.
        
        Same-slice: set Parameter.expr = "a*driver_key + b" with vary=True (algebraic)
        Cross-slice: depends on mode
          - Sequential (vary=False): compute value and freeze
          - Joint (vary=True): set expr and vary=True
        
        This method is called by FitOrchestrator for each target ParamRef that
        has this constraint rule.
        """

        target_pref = self.target_pref
        target_key = fmt_pref(target_pref)
        expr_lmfit = self.to_lmfit_expr()

        if target_key not in params:
            return

        p_tgt = params[target_key]

        same_slice = int(self.driver_pref.slice_id) == int(target_pref.slice_id)

        # --- Internal (same-slice): prefer algebraic expr if driver exists in params ---
        if same_slice:
            drv_key = fmt_pref(self.driver_pref)

            if drv_key in params:
                # Dependent parameter: express algebraic dependency.
                # Target should NOT be an independent DOF.
                
                p_tgt.set(expr=expr_lmfit, vary=True)
                return

            # Fallback: driver param not present -> numeric freeze using registry
            info = registry.get(self.driver_pref)
            if not info or "value" not in info:
                raise ValueError(f"Missing driver value for internal linear constraint: {fmt_pref(self.driver_pref)}")
            drv_val = float(info["value"])
            val = float(self.a) * drv_val + float(self.b)
            p_tgt.set(value=float(val), expr="", vary=False)
            return

        # --- External (cross-slice)
        if not allow_external:
            raise ValueError(
                f"Cannot apply cross-slice constraint on {target_key} when allow_external=False"
            )
        
        info = registry.get(self.driver_pref)
        if not info or "value" not in info:
            raise ValueError(f"Missing external driver value for {fmt_pref(self.driver_pref)}")

        drv_val = float(info["value"])
        val = float(self.a) * drv_val + float(self.b)

        if not vary: # Sequential mode: freeze with computed value. Clear expr.
            p_tgt.set(value=float(val), expr="", vary=False)
        else:  # Joint mode: use algebraic expression (cross-slice)            
            p_tgt.set(expr=expr_lmfit, vary=True)

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
        A: float = 1.0,
        C: float = 0.0,
        time_s: Optional[float] = None,
        T_number: Optional[float] = None,
        T_name: Optional[str] = None,
        expr_txt: str = "",
        enabled: bool = True
    ):
        """
        Initialize a relaxation exponential constraint.
        
        Args:
            target_pref: Parameter being constrained
            driver_pref: Parameter driving the decay
            A: Amplitude multiplicative factor
            C: Baseline offset
            time_s: Explicit time value (overrides T_name lookup)
            T_number: Decay time constant (seconds) can be None if T_name used
            T_name: Name of time constant can be None if T_seconds used
            enabled: Whether this constraint is active
        """
        self.target_pref = target_pref
        self.driver_pref = driver_pref
        self.A = float(A)
        self.C = float(C)
        self.time_s = float(time_s) if time_s is not None else None
        self.T_number = float(T_number) if T_number is not None else None
        self.T_name = T_name
        self.enabled = bool(enabled)
        self.expr_txt = str(expr_txt or "")
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.RELAX_DECAY
    
    @staticmethod
    def _split_left_mul(left: str) -> tuple[float, str]:
        """
        Parse the left-of-exp multiplicative chain, e.g.:
          "3*s1_p0_amp*" or "s1_p0_amp*3*" or "2 * s1_p0_amp * 0.5 *"
        Returns (A, driver_txt).
        """
        # Remove trailing '*' if present and strip whitespace
        s = left.strip()
        if s.endswith("*"):
            s = s[:-1].strip()
    
        # Split by '*' and clean tokens
        parts = [p.strip() for p in s.split("*") if p.strip()]    
    
        A = 1.0
        driver_parts: list[str] = []
    
        for p in parts:
            # accept plain floats like "3", "0.5", "1e-3"
            if is_float(p):
                A *= float(p)
            else:
                driver_parts.append(p)
    
        if len(driver_parts) != 1:
            raise ValueError(f"Expected exactly one driver term before exp(...), got: {driver_parts!r} from '{left}'")
    
        return A, driver_parts[0]

    @classmethod
    def parse_rhs(cls, UI_rhs_raw: str) -> Dict: #
        """
            parse right-hand side (rhs) like:
            "s15_p1_amp*1*exp(-0.1/T_name + 2)<br>"
            return dict with keys of A, driver_txt, C, t_override, T_name, T_number
        """
        out = {}
        raw = (UI_rhs_raw or "").strip()
        lower = raw.lower()   
        exp_idx = lower.find("exp(")
        if exp_idx == -1:
            raise ValueError(f"Malformed exponential constraint expression: '{UI_rhs_raw}'")

        left = raw[:exp_idx-1] # Left of *exp, e.g. 's15_p1_amp*1'
        right = raw[exp_idx+4:]  # Right of ( to end. remove the '*exp('

        # parse left side to get A and driver_txt
        A, driver_txt = RelaxDecayConstraint._split_left_mul(left)
        
        # parse right side to get t, T_name or T, C
        s = right.replace(")", "")
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
        
        out = {"A": A, "driver_txt": driver_txt, "C": C}
        
        # main should look like: -0.1/T_name   or  -0.1/0.035
        if not main.startswith("-"):
            raise ValueError(f"Expected '-' at start of exponential argument in '{UI_rhs_raw}'")
        if main.startswith("-"):
            main_k = main
        if "/" not in main_k:
            raise ValueError(f"Expected form like '-k/T_name' in '{UI_rhs_raw}'")
        t_part, T_part = main_k.split("/", 1)
        
        # add t_override
        t_val = float(t_part[1:])  # skip leading '-'
        out["t_override"] = t_val
        
        # T_part can be numeric or name
        if is_float(T_part):
            # numeric T
            # original exp() used "T"
            out["T_number"] = float(T_part)
        else:
            out["T_name"] = T_part

        return out
    
    @classmethod
    def parse_UItext(cls, UI_txt: str) -> Dict:
        t = (UI_txt or "").strip()
        low = t.lower()

        # Case 0: KV-style exp(...)
        if low.startswith("exp(") and ("a=" in low or "t=" in low or "t_name=" in low or "c=" in low):
            inner = t[t.find("(")+1 : t.rfind(")")]  # content inside exp(...)
            kv = LinearConstraint.parse_kv(inner)

            A = float(kv.get("A", 1.0))
            C = float(kv.get("C", 0.0))
            driver_txt = str(kv.get("driver", ""))

            # time
            t_override = kv.get("t", None)
            t_override = float(t_override) if t_override is not None and str(t_override).strip() != "" else None

            # T
            T_name = kv.get("T_name", None)
            T_number = kv.get("T", None)  # allow "T" as numeric seconds
            T_number = float(T_number) if T_number is not None and str(T_number).strip() != "" else None
            if T_name is not None and str(T_name).strip() == "":
                T_name = None

            return {
                "A": A,
                "driver_txt": driver_txt,
                "C": C,
                "t_override": t_override,
                "T_name": str(T_name) if T_name is not None else None,
                "T_number": T_number,
            }

        # Case 1: RHS-only (no "=" anywhere)
        if "=" not in t:
            return cls.parse_rhs(t)

        # Case 2: full assignment lhs = rhs  (but NOT kv exp(...))
        # (basic guard: if it starts with exp(, we would have matched above)
        lhs, rhs = t.split("=", 1)
        rhs = rhs.strip()
        return cls.parse_rhs(rhs)

        
    @staticmethod
    def UItext_to_expr(UItext) -> str:
        """Convert UI expression text into standard RHS expression for UI display and lmfit."""
        out = RelaxDecayConstraint.parse_UItext(UItext)
        A = float(out.get("A", 1.0))
        drv = out.get("driver_txt", None)
        C = float(out.get("C", 0.0))
        t_override = out.get("t_override", None)
        T_name = out.get("T_name", None)
        T_number = out.get("T_number", None)

        if A == 1.0 and C == 0.0:
            return f'{drv}*exp(-{t_override}/{T_number})' if T_number is not None else f'{drv}*exp(-{t_override}/{T_name})'
        if A != 1.0 and C == 0.0:
            return f'{A}*{drv}*exp(-{t_override}/{T_number})' if T_number is not None else f'{A}*{drv}*exp(-{t_override}/{T_name})'
        if C != 0.0:
            return f'{A}*{drv}*exp(-{t_override}/{T_number}) + {C}' if T_number is not None else f'{A}*{drv}*exp(-{t_override}/{T_name}) + {C}'
    
    def to_display_expr(self) -> str:
        """Safely call expr_txt or construct expr from self if expr_txt is none."""
        s = (self.expr_txt or "").strip()
        if s:
            return self.expr_txt

        drv = fmt_pref(self.driver_pref)
        A = float(self.A)
        C = float(self.C)
        t_override = self.time_s if self.time_s is not None else "t"
        T_name = self.T_name
        T_number = self.T_number

        if A == 1.0 and C == 0.0:
            return f'{drv}*exp(-{t_override}/{T_number})' if T_number is not None else f'{drv}*exp(-{t_override}/{T_name})'
        if A != 1.0 and C == 0.0:
            return f'{A}*{drv}*exp(-{t_override}/{T_number})' if T_number is not None else f'{A}*{drv}*exp(-{t_override}/{T_name})'
        if C != 0.0:
            return f'{A}*{drv}*exp(-{t_override}/{T_number}) + {C}' if T_number is not None else f'{A}*{drv}*exp(-{t_override}/{T_name}) + {C}'

    def to_lmfit_expr(self) -> str:
        """Return lmfit-compatible expression string."""
        display_text = self.to_display_expr()
        if self.T_number:
            lmfit_text = display_text
        if self.T_name:
            lmfit_text = display_text.replace(f"{self.T_name}", f"k__{self.T_name}")
        return lmfit_text
    
    @classmethod
    def infer_driver_from_expr(cls, expr_txt: str) -> Optional[str]:
        out = cls.parse_UItext(expr_txt)                
        return out.get("driver_txt")
    
    @classmethod
    def interpret_ui(
        cls,
        *,
        target_pref: "ParamRef",
        driver_txt: str,
        expr_txt: str,
        enabled: bool,
        ctx: "ParseContext",
    ) -> Optional["RelaxDecayConstraint"]:
        """Parse UI text, resolve time and T parameters, convert T→k, and call create_rule."""
        
        out = cls.parse_UItext(expr_txt)

        # Extract components (with defaults)
        A = float(out.get("A", 1.0))
        C = float(out.get("C", 0.0))
        T_number = out.get("T_number")     # numeric T in seconds (or None)
        T_name_raw = out.get("T_name")     # registry key (or None)
        T_name = (T_name_raw or "").strip() or None
        t_override = out.get("t_override") # optional numeric time override (or None)
        expr_driver_txt = (out.get("driver_txt", "") or "").strip()

        # ---- driver resolution (UI cell wins; else infer from expr) ----
        driver_pref = None
        ui_driver_txt = (driver_txt or "").strip()
        if ui_driver_txt:
            try:
                driver_pref = txt_to_pref(ui_driver_txt)
            except Exception:
                driver_pref = None

        if driver_pref is None and expr_driver_txt:
            try:
                driver_pref = txt_to_pref(expr_driver_txt)
            except Exception:
                driver_pref = None

        if driver_pref is None:
            raise ValueError("RelaxDecayConstraint requires a driver parameter.")

        # ---- time resolution (override wins; else metadata) ----
        time_s = ctx.resolve_time_s(slice_id=target_pref.slice_id, t_override_s=t_override)

        # Basic input sanity: forbid both numeric and named T
        if (T_number is not None) and (T_name is not None):
            raise ValueError("Provide either numeric T or T_name, not both.")
        if (T_number is None) and (T_name is None):
            raise ValueError("RelaxDecay requires either numeric T or T_name.")
        if T_number is not None:
            T_number = float(T_number)
            if T_number <= 0:
                raise ValueError(f"Numeric T must be > 0; got {T_number}.")

        # Optional: ensure registry row exists early (UI feedback), but still do NOT resolve k
        if T_name is not None:
            _ = ctx.get_T_record(T_name)  # raises if unknown

        return cls.create_rule(
            target_pref=target_pref,
            driver_pref=driver_pref,
            A=A,
            C=C,
            time_s=time_s,
            T_name=T_name,   # None if numeric T was used; registry T_name if used
            T_number=T_number,
            enabled=enabled,
            expr_txt=expr_txt,
        )
        
    @classmethod
    def create_rule(
        cls,
        *,
        target_pref: "ParamRef",
        driver_pref: "ParamRef",
        A: float = 1.0,
        C: float = 0.0,
        time_s: float,
        T_name: Optional[str] = None,
        T_number: Optional[float] = None,
        expr_txt: str = "",
        enabled: bool = True,
    ) -> "RelaxDecayConstraint":
        """Construct RelaxDecayConstraint from fully-resolved domain parameters.
        
        Args:
            time_s: Time value in seconds for this slice
            A: Amplitude coefficient
            C: Constant offset
            T_number: Optional numeric time constant for reference
            T_name: Optional registry key for time constant"""
        # Convert k back to T_number for storage if needed

        return cls(
            target_pref=target_pref,
            driver_pref=driver_pref,
            A=A,
            C=C,
            time_s=time_s,
            T_number=T_number,
            T_name=T_name,
            expr_txt=expr_txt,
            enabled=enabled,
        )
    
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
        if self.T_number is not None:
            try:
                T_val = float(self.T_number)
                if T_val <= 0.0:
                    errors.append(
                        f"Exponential constraint on {fmt_pref(target_pref)}: "
                        f"time constant T={T_val} must be positive"
                    )
            except (ValueError, TypeError):
                errors.append(
                    f"Exponential constraint on {fmt_pref(target_pref)}: "
                    f"time constant T={self.T_number} is not a valid number"
                )
        
        # Check time availability (either explicit or from slice_states)
        if self.time_s is None and self.T_name is None:
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
                    f"no time value available (no time_s or T_name, and slice_states has no t_f1)"
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
            *,
            params: Dict[str, Any],
            registry: Dict[ParamRef, Dict[str, Any]],
            allow_external: bool = False,
            vary: bool = True,
            Tseed_registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
        ) -> None:
            """
            Apply exponential decay constraint to lmfit.
            
            Same-slice: SUPPORTED for testing. Warn user. 
            (relaxation constraints are inherently cross-slice, depending on time vectors)
            Cross-slice: depends on mode
              - Sequential (vary=False): compute value using T_number seed, freeze target. T_number (algebraic expr not used is not supported)
              - Joint (vary=True): set algebraic expr with shared T parameter if T_name or T_number provided.
            
            Key difference from LINEAR:
              - Requires time data (t) per slice
              - Requires decay constant (T): numeric (T_number) or shared parameter (T_name)
              - For joint mode: would need T_name added to params
            """
            target_pref = self.target_pref
            target_key = fmt_pref(target_pref)            
            if target_key not in params:
                return
            #get necessary values from rule
            p_tgt = params[target_key]

            drv_key = fmt_pref(self.driver_pref)
            t_val = self.time_s
            T_val = float(self.T_number) if self.T_number is not None else self.T_name
            expr_lmfit = self.to_lmfit_expr()

            # Determine if same-slice or cross-slice         
            same_slice = int(self.driver_pref.slice_id) == int(target_pref.slice_id)
            
            # --- Internal (same-slice): allow for flexibility but warn user ---
            # Relaxation constraints are inherently cross-slice (time-dependent)
            # Only accept T_number (numeric) for same-slice use

            if same_slice:
                log.warning(
                    f"RELAX_DECAY constraint on {target_key} is inherently time-dependent and require cross-slice driver but uses same-slice driver."
                    f"Consider using LINEAR constraint instead for intra-slice relationships."
                )              
                if drv_key in params:
                    if t_val is None:
                        raise ValueError(
                            f"RELAX_DECAY constraint on {target_key}: no time value available"
                        )                                                   
                    # Set algebraic dependency
                    expr_lmfit = self.to_lmfit_expr()
                    p_tgt.set(expr=expr_lmfit, vary=True)
                    return
                
                # Fallback: driver not present → numeric freeze using registry
                info = registry.get(self.driver_pref)
                if not info or "value" not in info:
                    raise ValueError(
                        f"Missing driver value for internal RELAX_DECAY constraint: {fmt_pref(self.driver_pref)}"
                    )               
                drv_val = float(info["value"])
                if t_val is None:
                    raise ValueError(
                        f"RELAX_DECAY constraint on {target_key}: no time value available"
                    )
                if is_float(T_val) and T_val >= 0:
                    T_val = float(T_val)
                    exp_term = np.exp(-t_val / T_val)
                    val = drv_val * self.A * exp_term + self.C                
                    p_tgt.set(value=float(val), expr=expr_lmfit, vary=True)
                return
            
            # --- External (cross-slice) ---
            if not allow_external:
                raise ValueError(
                    f"Cannot apply cross-slice RELAX_DECAY constraint on {target_key} when allow_external=False"
                )
            
            if not vary:
                # Sequential mode: freeze with computed value
                # 2 cases T_number (numeric) or T_name (not supported)
                # Get driver value from registry
                info = registry.get(self.driver_pref)
                if not info or "value" not in info:
                    raise ValueError(f"Missing external driver value for {fmt_pref(self.driver_pref)}")
                drv_val = float(info["value"])
                # Compute: driver * A * exp(-t / T) + C
                if is_float(T_val) and T_val > 0: #T_number case
                    T_val = float(T_val)
                    exp_term = np.exp(-t_val / T_val)
                    val = drv_val * self.A * exp_term + self.C                
                    p_tgt.set(value=float(val), expr=expr_lmfit, vary=False)              
                else: #T_val is not numeric but T_name
                    raise ValueError(f"Sequential fitting does not support {T_val}")
            else:
                # Joint mode: algebraic expression with shared T parameter
                # T_number -> RelaxDecayConstraints acts like LinearConstraint with a = A*exp(-t/T_number)
                if self.T_name or self.T_number is not None:
                    p_tgt.set(expr=expr_lmfit, vary=True)
                if self.T_name: # T_name -> convert T_name to k_name and inject k_name into parameter if not already present
                    if Tseed_registry is None:
                        raise ValueError(f"RELAX_DECAY uses T_name='{self.T_name}' but tseed_registry was not provided.")
                    # Create or update k__T in params with bounds/vary from registry
                    ensure_k_param_in_lmfit(
                        params=params,
                        T_name=self.T_name,
                        Tseed_registry=Tseed_registry,
                    )

class ConstraintRuleFactory:
    """
    Factory for creating constraint rules from serialized representations.
    Useful for loading/saving constraints to file and creating rules from user input.
    """
    
    @staticmethod
    def to_dict(self, store) -> Dict:
        """Serialize constraint rule to dictionary."""
        pass

    def from_dict(self, dict) -> ConstraintStore:
        """Parse dictionary to constraint rule."""
        pass


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
    
    elif link_type == "RELAX_DECAY":
        return RelaxDecayConstraint(
            target_pref=link_expr.target,
            driver_pref=link_expr.driver,
            T_number=link_expr.args.get("T", 1.0),
            A=link_expr.args.get("A", 1.0),
            C=link_expr.args.get("C", 0.0),
            time_s=link_expr.args.get("t_override"),
            T_name=link_expr.args.get("T_name"),
            k_seed=None,
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
            "T": rule.T_number,
            "A": rule.A,
            "C": rule.C
        }
        if rule.time_s is not None:
            args["t_override"] = rule.time_s
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
class LinkEngine:
    """
    Dependency-analysis engine that works with the *new* ConstraintStore (not LinkStore).

    Design goals:
      - UI-agnostic: no imports from nmrFit_v0 / Qt
      - Operates only on ParamRef + ConstraintRule via rule.driver_pref
      - Provides graph utilities needed by FitOrchestrator (external driver discovery,
        per-slice filtering, topo-ordering for future registry evaluation)
    """

    def rules_for_target_slice(
        self,
        *,
        store: "ConstraintStore",
        slice_id: int,
        enabled_only: bool = True,
    ) -> List[Tuple[ParamRef, "ConstraintRule"]]:
        """
        Return [(target_pref, rule), ...] whose *target* lives in slice_id.
        """
        out: List[Tuple[ParamRef, "ConstraintRule"]] = []
        for tgt, rule in store.all_constraints():
            if int(tgt.slice_id) != int(slice_id):
                continue
            if enabled_only and not getattr(rule, "enabled", True):
                continue
            out.append((tgt, rule))
        return out

    def get_external_driver_slices(
        self,
        *,
        store: "ConstraintStore",
        target_slice_id: int,
        enabled_only: bool = True,
    ) -> Set[int]:
        """
        Return the set of slice_ids that appear as *drivers* for constraints whose
        targets are in target_slice_id, excluding same-slice drivers.
        """
        out: Set[int] = set()
        for tgt, rule in self.rules_for_target_slice(
            store=store, slice_id=target_slice_id, enabled_only=enabled_only
        ):
            drv = getattr(rule, "driver_pref", None)
            if drv is None:
                continue
            ds = int(drv.slice_id)
            ts = int(tgt.slice_id)
            if ds != ts:
                out.add(ds)
        return out

    def topo_sort_paramrefs(
        self,
        *,
        store: "ConstraintStore",
        enabled_only: bool = True,
    ) -> List[ParamRef]:
        """
        Topologically sort ParamRefs using dependencies implied by rule.driver_pref.
        Nodes include enabled targets and any referenced drivers.

        Raises:
            ValueError if a cycle is detected.
        """
        edges: Dict[ParamRef, List[ParamRef]] = {}
        indeg: Dict[ParamRef, int] = {}

        for tgt, rule in store.all_constraints():
            if enabled_only and not getattr(rule, "enabled", True):
                continue
            indeg.setdefault(tgt, 0)
            drv = getattr(rule, "driver_pref", None)
            if drv is None:
                continue
            edges.setdefault(drv, []).append(tgt)
            indeg.setdefault(drv, 0)
            indeg[tgt] = indeg.get(tgt, 0) + 1

        q: List[ParamRef] = [n for n, k in indeg.items() if k == 0]
        ordered: List[ParamRef] = []
        while q:
            u = q.pop()
            ordered.append(u)
            for v in edges.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(ordered) != len(indeg):
            raise ValueError("Link/constraint cycle detected among enabled rules.")
        return ordered


@dataclass
class FitOrchestrator:
    """
    Orchestrates constraint validation and application for fitting.
    
    Responsibilities:
    - Validate all constraints before fitting
    - Apply constraints to lmfit.Parameters in correct order
    - Handle cross-slice dependencies
    
    Does NOT:
    - Execute fits (that's lmfit's job)
    - Parse UI text (that's dispatch_rule_from_ui's job)
    - Branch on constraint types (that's ConstraintRule.apply_to_lmfit's job)
    """
    constraint_store: ConstraintStore
    link_engine: LinkEngine = field(default_factory=LinkEngine)
    Tseed_registry: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def rules_for_target_slice(
        self, *, slice_id: int, enabled_only: bool = True
    ) -> List[Tuple[ParamRef, "ConstraintRule"]]:
        return self.link_engine.rules_for_target_slice(
            store=self.constraint_store, slice_id=slice_id, enabled_only=enabled_only
        )

    def get_external_driver_slices(
        self, *, target_slice_id: int, enabled_only: bool = True
    ) -> Set[int]:
        return self.link_engine.get_external_driver_slices(
            store=self.constraint_store, target_slice_id=target_slice_id, enabled_only=enabled_only
        )
    
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
    
    @staticmethod
    def build_registry_for_slice(
        *,
        slice_id: int,
        slice_states: Dict[int, Any],
        peaks_override: Optional[List[Any]] = None,
    ) -> Dict["ParamRef", Dict[str, Any]]:
        """
        Build a numeric registry dict for one slice. 
        Bridge between Peaks and ConstraintRules.
        Registry contains a mapping:
            ParamRef -> {"value": float} (Peak parameter values)

        Registry also contains a cross-slice driver if any such driver is needed.

        Notes:
        - This function is UI-agnostic.
        - It reads peak values from either:
            (a) peaks_override (authoritative for current slice), or
            (b) slice_states[slice_id].peaks
        - Bounds are NOT included (bounds belong in FitContext._build_params).
        """
        sid = int(slice_id)
        reg: Dict["ParamRef", Dict[str, Any]] = {}

        # Decide source of peaks
        peaks = peaks_override
        if peaks is None:
            st = (slice_states or {}).get(sid, None)
            peaks = getattr(st, "peaks", None) if st is not None else None

        if not peaks:
            return reg

        # Map ParamRef.name -> Peak attribute
        name_to_attr = {"pos": "pos", "amp": "amp", "lor": "lor_hz", "gauss": "gauss_disp",}

        for pid, pk in enumerate(peaks):
            pid = int(pid)
            for pname, attr in name_to_attr.items():
                try:
                    val = float(getattr(pk, attr))
                except Exception:
                    # Defensive: skip missing/invalid values rather than crashing the fit
                    continue
                pref = ParamRef(slice_id=sid, peak_id=pid, name=pname)
                reg[pref] = {"value": val}

        return reg
    
    @staticmethod
    def build_registry_for_slices(
        *,
        slice_ids: Iterable[int],
        slice_states: Dict[int, Any],
        peaks_overrides: Optional[Dict[int, List[Any]]] = None,
    ) -> Dict["ParamRef", Dict[str, Any]]:
        """
        Build a merged numeric registry for multiple slices.
        Args:
            slice_ids:
                Slices to include in the registry.
            slice_states:
                {slice_id -> SliceFitState-like} object whose .peaks holds Peak objects.
            peaks_overrides:
                Optional {slice_id -> peaks_list}. Use this when the current slice’s
                peaks are "live" in the UI/model and may not match slice_states[sid].peaks.

        Returns:
            A single merged registry dict[ParamRef] -> {"value": float, "fixed": bool}.
            If the same ParamRef is encountered twice, later entries overwrite earlier ones.
        """
        reg: Dict["ParamRef", Dict[str, Any]] = {}
        overrides = peaks_overrides or {}

        for sid in slice_ids:
            sid = int(sid)
            part = FitOrchestrator.build_registry_for_slice(
                slice_id=sid,
                slice_states=slice_states,
                peaks_override=overrides.get(sid),
            )
            reg.update(part)

        return reg
    
    def _seed_external_drivers_into_registry(
        self,
        *,
        registry: Dict[ParamRef, Dict[str, Any]],
        slice_states: Dict[int, Any],
        target_slice_id: int,
        enabled_only: bool = True,
        strict: bool = True,
    ) -> Set[ParamRef]:
        """
        For enabled constraints targeting target_slice_id whose driver lives in another slice,
        inject the driver's numeric value into registry.

        Returns:
          Set of external (cross-slice) driver ParamRefs required by constraints targeting this slice.
        """
        sid = int(target_slice_id)
        missing: List[Tuple[ParamRef, str]] = []
        external: Set[ParamRef] = set()

        # 1) Collect external drivers needed for rules targeting this slice
        for tgt, rule in self.constraint_store.all_constraints():
            if enabled_only and not getattr(rule, "enabled", True):
                continue
            if int(tgt.slice_id) != sid:
                continue

            drv = getattr(rule, "driver_pref", None)
            if drv is None:
                continue

            if int(drv.slice_id) != int(tgt.slice_id):
                external.add(drv)

        if not external:
            return set()

        # 2) Canonical mapping name -> Peak attribute
        name_to_attr = {"pos": "pos", "amp": "amp", "lor": "lor_hz", "gauss": "gauss_disp",}

        # 3) Seed values
        for drv in external:
            # Preserve any caller-provided value
            if drv in registry and "value" in (registry.get(drv) or {}):
                continue

            ds = int(drv.slice_id)
            st = (slice_states or {}).get(ds)
            if st is None:
                missing.append((drv, f"slice_states[{ds}] not found"))
                continue

            peaks = getattr(st, "peaks", None)
            if not peaks:
                missing.append((drv, f"slice_states[{ds}].peaks missing/empty"))
                continue

            try:
                pk = peaks[int(drv.peak_id)]
            except Exception:
                missing.append((drv, f"peak_id {drv.peak_id} not present in slice {ds} (len={len(peaks)})"))
                continue

            pname = getattr(drv, "name", None)
            attr = name_to_attr.get(pname)
            if attr is None:
                missing.append((drv, f"non-canonical ParamRef.name '{pname}' (expected one of {sorted(name_to_attr)})"))
                continue

            try:
                val = float(getattr(pk, attr))
            except Exception as ex:
                missing.append((drv, f"value read failed for attr '{attr}': {ex}"))
                continue

            registry[drv] = {"value": val}

        if strict and missing:
            lines = [f"- driver {fmt_pref(drv)}: {why}" for drv, why in missing]
            raise RuntimeError("Cross-slice driver seeding failed:\n" + "\n".join(lines))

        return external


    def apply_constraints_to_lmfit(
        self,
        *,
        params,  # lmfit.Parameters (dict-like)
        registry: Dict[ParamRef, Dict[str, Any]],  # {'value': float}
        slice_states: Dict[int, Any],
        slice_id: int,
        allow_external: bool,
        vary: bool,
        strict_external_seed: bool = True,
        Tseed_registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        """
        Apply enabled constraints targeting slice_id.

        If allow_external is True:
          - seed external driver values from slice_states into registry (unless already provided)

        If allow_external is False:
          - raise ValueError if any cross-slice drivers are required for this slice

        Args:
          params: lmfit.Parameters
          registry: ParamRef -> {'value': float}
          slice_states: slice_id -> SliceFitState-like (must provide .peaks for external seeding)
          slice_id: apply only constraints whose target_pref.slice_id == slice_id
          allow_external: whether cross-slice drivers are allowed
          vary: mode hint (False sequential, True joint)
          strict_external_seed: whether missing external values raise (True) or warn+skip (False)
        """
        sid = int(slice_id)

        # Targets in this run (only the current slice)
        targets_in_run = {p for p in registry.keys() if int(p.slice_id) == sid}

        rules_internal: List[Tuple[ParamRef, Any]] = [] # (target, rule)
        rules_external: List[Tuple[ParamRef, Any]] = [] # (target, rule)

        # Collect enabled rules for targets present in this run
        for tgt in targets_in_run:
            rule = self.constraint_store.get_constraint(tgt)
            if rule is None or not getattr(rule, "enabled", True):
                continue

            drv = getattr(rule, "driver_pref", None)
            if drv is None:
                # If you ever allow driverless constraints later, handle here.
                # For now, skip/raise depending on policy.
                continue

            if int(drv.slice_id) == int(tgt.slice_id):
                rules_internal.append((tgt, rule))
            else:
                rules_external.append((tgt, rule))

        if not rules_internal and not rules_external:
            return

        # External drivers: enforce allow_external
        if rules_external and not allow_external:
            driver_slices = sorted({int(getattr(rule, "driver_pref").slice_id) for _, rule in rules_external})
            raise ValueError(
                f"Cross-slice constraints detected for slice {sid}: driver slices={driver_slices}. "
                "Call with allow_external=True (and provide slice_states), or use Fit Selected."
            )

        # Seed external driver values into registry for apply_to_lmfit call later to calculate driver values.
        external_drivers: Set[ParamRef] = set()
        if rules_external and allow_external:
            external_drivers = self._seed_external_drivers_into_registry(
                registry=registry,
                slice_states=slice_states,
                target_slice_id=sid,
                strict=strict_external_seed,
            )

        # Apply INTERNAL rules first (so same-slice expr wiring happens before numeric freezes)
        for tgt, rule in rules_internal:
            try:
                # warn+override: internal driver must be able to vary if use expr
                drv = getattr(rule, "driver_pref", None)
                if drv is not None:
                    drv_key = fmt_pref(drv)
                    if drv_key in params and getattr(params[drv_key], "vary", True) is False:
                        log.warning(
                            "Constraint override: setting driver %s vary=True (required by internal constraint).",
                            drv_key,
                        )
                        params[drv_key].vary = True
                tgt_key = fmt_pref(tgt)
                if tgt_key in params and getattr(params[tgt_key], "vary", True) is False:                    
                    log.warning(
                        "Constraint override: setting target %s vary=True (required by internal constraint).",
                        tgt_key,
                    )
                    params[tgt_key].vary = True

                rule.apply_to_lmfit(
                    params=params,
                    registry=registry,
                    allow_external=False,   # internal
                    vary=vary,              # still pass through; rule may use it
                    Tseed_registry=Tseed_registry,
                )
            except Exception as e:
                log.warning("Failed to apply internal constraint for %s: %s", fmt_pref(tgt), e)

        # Apply EXTERNAL rules
        for tgt, rule in rules_external:
            try:
                if vary:
                    # Joint mode but cross-slice expr not supported in current architecture:
                    log.warning(
                        "Cross-slice constraint in joint mode is applied numerically (frozen) for %s. "
                        "True joint cross-slice constraints require shared/global lmfit Parameters.",
                        fmt_pref(tgt),
                    )

                rule.apply_to_lmfit(
                    params=params,
                    registry=registry,
                    allow_external=True,   # external
                    vary=vary,
                    Tseed_registry=self.Tseed_registry,
                )
            except Exception as e:
                log.warning("Failed to apply external constraint for %s: %s", fmt_pref(tgt), e)


@dataclass(frozen=True)
class ParseContext:
    """
    Creation-time context for interpreting UI constraint text.

    Design goals:
      - UI-agnostic: no Qt types.
      - Read-only: constructed by the GUI/controller at commit time.
      - Wraps existing providers (Tseed_registry, time metadata) without duplicating ownership.

    Typical usage:
      - LinearConstraint ignores ctx.
      - RelaxDecayConstraint uses ctx to resolve:
          (a) per-slice time t
          (b) T seeds/bounds by T_name
          (c) conversions T -> k and bounds mapping
    """
    # --- Time provider ---
    # Return time in seconds for a given slice_id, or None if unavailable.
    time_for_slice: Callable[[int], Optional[float]]

    # --- Tseed_registry (owned by nmrFit MainWindow) ---
    # Expected record keys: "T_seed_s", "T_lo_s", "T_hi_s", "fixed"
    Tseed_registry: Mapping[str, Mapping[str, Any]]
    ensure_tseed_row: Optional[Callable[[str], None]] = None

    # Optional: enforce positive values
    require_positive_T: bool = True
    require_positive_time: bool = False #allow negative relative times. can be switched on if needed

    # Numerical guard to avoid division blow-ups
    eps: float = 1e-15

    # -----------------------------
    # Time resolution
    # -----------------------------
    def resolve_time_s(self, *, slice_id: int, t_override_s: Optional[float] = None) -> float:
        """
        Resolve time (seconds) using the precedence:
          1) explicit override (typed by user)
          2) metadata time_for_slice(slice_id)

        Raises ValueError if missing or invalid.
        """
        if t_override_s is not None:
            t = float(t_override_s)
        else:
            t = self.time_for_slice(int(slice_id))
            if t is None:
                raise ValueError(f"No time value available for slice {slice_id}.")
            t = float(t)

        if self.require_positive_time and t < 0:
            raise ValueError(f"Time must be >= 0 s; got t={t} for slice {slice_id}.")
        return t

    # -----------------------------
    # T resolution (seed/bounds/fix)
    # -----------------------------
    def get_T_record(self, T_name: str) -> Mapping[str, Any]:
        """
        Return the registry record for T_name, or raise if missing.
        Tseed_registry: dict {T_name: {T_seed_s: value, T_lo_s, T_hi_s, fixed}}
        """
        name = (T_name or "").strip()
        if not name:
            raise ValueError("Empty T_name.")
        rec = self.Tseed_registry.get(name, None)
        
        if rec is None and self.ensure_tseed_row is not None:
        # Auto-create + UI reflect (MainWindow-owned)
            self.ensure_tseed_row(name)
            rec = self.Tseed_registry.get(name)
        
        if rec is None:
            raise ValueError(f"Unknown T_name '{name}'. Add it to the T seeds table.")
        return rec

    def resolve_T_seed_s(self, *, T_name: str) -> float:
        """
        Read T_seed_s from the registry. Raises if missing or invalid.
        """
        rec = self.get_T_record(T_name)
        T = rec.get("T_seed_s", None)
        if T is None:
            raise ValueError(f"T_seed_s is not set for '{T_name}'.")
        T = float(T)
        if self.require_positive_T and T <= 0:
            raise ValueError(f"T must be > 0 s; got T={T} for '{T_name}'.")
        return T

    def resolve_T_bounds_s(self, *, T_name: str) -> ParamBounds:
        """
        Read T_lo_s / T_hi_s from the registry. Missing values become None.
        Validates lo<=hi if both are set.
        """
        rec = self.get_T_record(T_name)
        lo = rec.get("T_lo_s", None)
        hi = rec.get("T_hi_s", None)

        lo_f = None if lo is None else float(lo)
        hi_f = None if hi is None else float(hi)

        if self.require_positive_T:
            if lo_f is not None and lo_f <= 0:
                raise ValueError(f"T_lo_s must be > 0 for '{T_name}'; got {lo_f}.")
            if hi_f is not None and hi_f <= 0:
                raise ValueError(f"T_hi_s must be > 0 for '{T_name}'; got {hi_f}.")

        if (lo_f is not None) and (hi_f is not None) and (lo_f > hi_f):
            raise ValueError(f"Invalid bounds for '{T_name}': lo({lo_f}) > hi({hi_f}).")

        return ParamBounds(lo=lo_f, hi=hi_f)

    def is_T_fixed(self, *, T_name: str) -> bool:
        rec = self.get_T_record(T_name)
        return bool(rec.get("fixed", False))

    # -----------------------------
    # Conversions: T <-> k
    # -----------------------------
    def T_to_k(self, T_s: float) -> float:
        """
        Convert T (seconds) to k (1/seconds): k = 1/T.
        """
        T = float(T_s)
        if self.require_positive_T and T <= 0:
            raise ValueError(f"T must be > 0 to convert to k; got T={T}.")
        if abs(T) < self.eps:
            raise ValueError(f"T too small to convert safely (|T|<{self.eps}).")
        return 1.0 / T

    def k_to_T(self, k: float) -> float:
        """
        Convert k (1/seconds) to T (seconds): T = 1/k.
        """
        kk = float(k)
        if abs(kk) < self.eps:
            raise ValueError(f"k too small to convert safely (|k|<{self.eps}).")
        T = 1.0 / kk
        if self.require_positive_T and T <= 0:
            raise ValueError(f"Derived T must be > 0; got T={T}.")
        return T

    def T_bounds_to_k_bounds(self, T_bounds: ParamBounds) -> ParamBounds:
        """
        Map bounds on T to bounds on k=1/T.

        If T in [lo, hi] with lo>0:
          k in [1/hi, 1/lo]
        """
        loT, hiT = T_bounds.lo, T_bounds.hi

        loK = None
        hiK = None

        # hiT gives loK
        if hiT is not None:
            loK = self.T_to_k(float(hiT))

        # loT gives hiK
        if loT is not None:
            hiK = self.T_to_k(float(loT))

        # Validate ordering if both are set
        if (loK is not None) and (hiK is not None) and (loK > hiK):
            # This should not happen if T bounds were valid and monotonic,
            # but keep it defensive.
            raise ValueError(f"Derived k bounds invalid: lo({loK}) > hi({hiK}).")

        return ParamBounds(lo=loK, hi=hiK)

    # -----------------------------
    # High-level helper for RelaxDecay
    # -----------------------------
    def resolve_k_and_bounds(
        self,
        *,
        T_number_s: Optional[float],
        T_name: Optional[str],
    ) -> Tuple[float, Optional[ParamBounds], Optional[str], bool]:
        """
        Resolve k (1/s) and optional k-bounds from either numeric T_number_s or named T_name.

        Precedence:
          1) numeric T_number_s (if provided)
          2) T_name -> registry

        Returns:
          (k_value, k_bounds_or_None, resolved_T_name_or_None, is_fixed)

        Notes:
          - If numeric T is used, bounds come back as None.
          - If T_name is used, bounds are derived from T_lo_s/T_hi_s and mapped to k.
        """
        name = (T_name or "").strip() or None

        # Disallow ambiguous input
        if (T_number_s is not None) and (name is not None):
            raise ValueError("Provide either numeric T or T_name, not both.")

        if T_number_s is not None:
            T = float(T_number_s)
            k = self.T_to_k(T)
            return k, None, None, True   # recommend fixed=True for explicit numeric T

        if name is None:
            raise ValueError("RelaxDecay requires either numeric T or T_name.")

        T_seed = self.resolve_T_seed_s(T_name=name)
        T_bounds = self.resolve_T_bounds_s(T_name=name)
        k = self.T_to_k(T_seed)
        k_bounds = self.T_bounds_to_k_bounds(T_bounds) if T_bounds.is_set() else None
        fixed = self.is_T_fixed(T_name=name)
        return k, k_bounds, name, fixed


register_constraint_rule(ConstraintType.LINEAR, LinearConstraint)
register_constraint_rule(ConstraintType.RELAX_DECAY, RelaxDecayConstraint)
register_constraint_rule(ConstraintType.RELAX_GROWTH, RelaxDecayConstraint)