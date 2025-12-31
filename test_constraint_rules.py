"""
Unit tests for constraint_rules.py

Tests ConstraintRule hierarchy, validation, evaluation, and serialization.
"""

import unittest
import math
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from constraint_rules import (
    ConstraintType,
    ConstraintRule,
    LinearConstraint,
    RelaxDecayConstraint,
    ConstraintRuleFactory,
    LinkExpr_to_ConstraintRule,
    ConstraintRule_to_LinkExpr,
)
from nmrFit_v0 import ParamRef, Peak, LinkExpr, LinkType


# ============================================================================
# Test Fixtures
# ============================================================================

class TestConstraintRules(unittest.TestCase):
    """Test ConstraintRule hierarchy and subclasses."""

    def setUp(self):
        """Create common test fixtures."""
        # Create sample peaks
        self.peak_s0_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        self.peak_s0_p1 = Peak(pos=110.0, amp=30.0, lor_hz=1.5, gauss_disp=0.3)
        self.peak_s1_p0 = Peak(pos=105.0, amp=40.0, lor_hz=1.2, gauss_disp=0.4)
        
        # Create sample ParamRefs
        self.pref_s0_p0_amp = ParamRef(slice_id=0, peak_id=0, name="amp")
        self.pref_s0_p0_pos = ParamRef(slice_id=0, peak_id=0, name="pos")
        self.pref_s0_p1_amp = ParamRef(slice_id=0, peak_id=1, name="amp")
        self.pref_s0_p1_pos = ParamRef(slice_id=0, peak_id=1, name="pos")
        self.pref_s1_p0_amp = ParamRef(slice_id=1, peak_id=0, name="amp")
        
        # Create sample slice states
        self.slice_states = {
            0: {
                "peaks": [self.peak_s0_p0, self.peak_s0_p1],
                "time_seconds": None
            },
            1: {
                "peaks": [self.peak_s1_p0],
                "time_seconds": 0.5
            }
        }

    # ========================================================================
    # LinearConstraint Tests
    # ========================================================================

    def test_linear_constraint_type(self):
        """Test that LinearConstraint has correct constraint_type."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0
        )
        self.assertEqual(rule.constraint_type, ConstraintType.LINEAR)

    def test_linear_constraint_validate_success(self):
        """Test LinearConstraint.validate() with valid inputs."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        self.assertEqual(errors, [])

    def test_linear_constraint_validate_no_driver(self):
        """Test LinearConstraint.validate() with no driver (standalone constant)."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=None,
            a=0.0,
            b=5.0  # Just a constant
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=None,
            driver_pref=None,
            slice_states=self.slice_states
        )
        self.assertEqual(errors, [])

    def test_linear_constraint_validate_missing_driver(self):
        """Test LinearConstraint.validate() fails when driver is required but missing."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=None,  # Missing!
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("driver" in err.lower() for err in errors))

    def test_linear_constraint_validate_infinite_coefficients(self):
        """Test LinearConstraint.validate() rejects infinite coefficients."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=float('inf'),
            b=1.0
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        self.assertTrue(len(errors) > 0)

    def test_linear_constraint_evaluate_with_driver(self):
        """Test LinearConstraint.evaluate() with driver: target = 2*30 + 1 = 61."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0
        )
        value = rule.evaluate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        self.assertAlmostEqual(value, 61.0)

    def test_linear_constraint_evaluate_without_driver(self):
        """Test LinearConstraint.evaluate() without driver: just returns b."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=None,
            a=0.0,
            b=5.0
        )
        value = rule.evaluate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=None,
            driver_pref=None,
            slice_states=self.slice_states
        )
        self.assertAlmostEqual(value, 5.0)

    def test_linear_constraint_evaluate_negative_driver(self):
        """Test LinearConstraint.evaluate() with negative driver coefficient."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=-1.5,
            b=10.0
        )
        value = rule.evaluate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        # value = -1.5 * 30 + 10 = -35
        self.assertAlmostEqual(value, -35.0)

    # ========================================================================
    # RelaxDecayConstraint Tests
    # ========================================================================

    def test_relax_decay_constraint_type(self):
        """Test that RelaxDecayConstraint has correct constraint_type."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.0,
            C=0.0,
            time_seconds=0.5
        )
        self.assertEqual(rule.constraint_type, ConstraintType.RELAX_EXP)

    def test_relax_decay_constraint_validate_success(self):
        """Test RelaxDecayConstraint.validate() with valid inputs."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.0,
            C=0.0,
            time_seconds=0.5
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        self.assertEqual(errors, [])

    def test_relax_decay_constraint_validate_missing_driver(self):
        """Test RelaxDecayConstraint.validate() fails without driver."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.0,
            C=0.0,
            time_seconds=0.5
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=None,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        self.assertTrue(len(errors) > 0)

    def test_relax_decay_constraint_validate_negative_T(self):
        """Test RelaxDecayConstraint.validate() rejects T <= 0."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=-1.0,  # Invalid!
            A=1.0,
            C=0.0,
            time_seconds=0.5
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("T" in err for err in errors))

    def test_relax_decay_constraint_validate_no_time(self):
        """Test RelaxDecayConstraint.validate() fails when time not available."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.0,
            C=0.0,
            time_seconds=None,
            T_name=None  # No time source!
        )
        errors = rule.validate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states={0: {"peaks": [self.peak_s0_p0, self.peak_s0_p1], "time_seconds": None}}
        )
        self.assertTrue(len(errors) > 0)

    def test_relax_decay_constraint_evaluate(self):
        """Test RelaxDecayConstraint.evaluate(): driver * A * exp(-t/T) + C."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.0,
            C=0.0,
            time_seconds=0.5
        )
        value = rule.evaluate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        # value = 30 * 1 * exp(-0.5/1) + 0 = 30 * exp(-0.5)
        expected = 30.0 * math.exp(-0.5)
        self.assertAlmostEqual(value, expected, places=6)

    def test_relax_decay_constraint_evaluate_with_offset(self):
        """Test RelaxDecayConstraint.evaluate() with C offset."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.0,
            C=5.0,  # Add offset
            time_seconds=0.5
        )
        value = rule.evaluate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        # value = 30 * 1 * exp(-0.5/1) + 5
        expected = 30.0 * math.exp(-0.5) + 5.0
        self.assertAlmostEqual(value, expected, places=6)

    def test_relax_decay_constraint_evaluate_zero_time(self):
        """Test RelaxDecayConstraint.evaluate() at t=0 returns driver * A + C."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=2.0,
            C=1.0,
            time_seconds=0.0  # At t=0
        )
        value = rule.evaluate(
            target_peak=self.peak_s0_p0,
            target_pref=self.pref_s0_p0_amp,
            driver_peak=self.peak_s0_p1,
            driver_pref=self.pref_s0_p1_amp,
            slice_states=self.slice_states
        )
        # value = 30 * 2 * exp(0) + 1 = 60 + 1 = 61
        self.assertAlmostEqual(value, 61.0, places=6)

    # ========================================================================
    # ConstraintRuleFactory Tests
    # ========================================================================

    def test_factory_linear_constraint_round_trip(self):
        """Test LinearConstraint serialization → deserialization."""
        original = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.5,
            b=0.5,
            enabled=True
        )
        
        # Serialize
        data = ConstraintRuleFactory.to_dict(original)
        
        # Deserialize
        restored = ConstraintRuleFactory.from_dict(data)
        
        # Verify
        self.assertIsInstance(restored, LinearConstraint)
        self.assertEqual(restored.target_pref, original.target_pref)
        self.assertEqual(restored.driver_pref, original.driver_pref)
        self.assertAlmostEqual(restored.a, original.a)
        self.assertAlmostEqual(restored.b, original.b)
        self.assertEqual(restored.enabled, original.enabled)

    def test_factory_linear_constraint_no_driver(self):
        """Test LinearConstraint without driver (constant)."""
        original = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=None,
            a=0.0,
            b=10.0,
            enabled=True
        )
        
        data = ConstraintRuleFactory.to_dict(original)
        restored = ConstraintRuleFactory.from_dict(data)
        
        self.assertIsInstance(restored, LinearConstraint)
        self.assertIsNone(restored.driver_pref)
        self.assertAlmostEqual(restored.b, 10.0)

    def test_factory_relax_decay_constraint_round_trip(self):
        """Test RelaxDecayConstraint serialization → deserialization."""
        original = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=0.5,
            A=1.5,
            C=0.2,
            time_seconds=0.1,
            T_name=None,
            enabled=True
        )
        
        # Serialize
        data = ConstraintRuleFactory.to_dict(original)
        
        # Deserialize
        restored = ConstraintRuleFactory.from_dict(data)
        
        # Verify
        self.assertIsInstance(restored, RelaxDecayConstraint)
        self.assertEqual(restored.target_pref, original.target_pref)
        self.assertEqual(restored.driver_pref, original.driver_pref)
        self.assertAlmostEqual(restored.T_seconds, original.T_seconds)
        self.assertAlmostEqual(restored.A, original.A)
        self.assertAlmostEqual(restored.C, original.C)
        self.assertAlmostEqual(restored.time_seconds, original.time_seconds)
        self.assertEqual(restored.T_name, original.T_name)
        self.assertEqual(restored.enabled, original.enabled)

    def test_factory_relax_decay_constraint_with_T_name(self):
        """Test RelaxDecayConstraint with T_name instead of time_seconds."""
        original = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.0,
            C=0.0,
            time_seconds=None,
            T_name="T1_s",  # Reference global parameter
            enabled=True
        )
        
        data = ConstraintRuleFactory.to_dict(original)
        restored = ConstraintRuleFactory.from_dict(data)
        
        self.assertIsInstance(restored, RelaxDecayConstraint)
        self.assertEqual(restored.T_name, "T1_s")
        self.assertIsNone(restored.time_seconds)

    # ========================================================================
    # Migration Bridge Tests
    # ========================================================================

    def test_migration_link_expr_to_linear_constraint(self):
        """Test LinkExpr_to_ConstraintRule() for LINEAR type."""
        link_expr = LinkExpr(
            type=LinkType.LINEAR,
            target=self.pref_s0_p0_amp,
            driver=self.pref_s0_p1_amp,
            args={"a": 2.5, "b": 0.5},
            enabled=True
        )
        
        rule = LinkExpr_to_ConstraintRule(link_expr)
        
        self.assertIsInstance(rule, LinearConstraint)
        self.assertEqual(rule.target_pref, self.pref_s0_p0_amp)
        self.assertEqual(rule.driver_pref, self.pref_s0_p1_amp)
        self.assertAlmostEqual(rule.a, 2.5)
        self.assertAlmostEqual(rule.b, 0.5)
        self.assertEqual(rule.enabled, True)

    def test_migration_link_expr_to_relax_decay_constraint(self):
        """Test LinkExpr_to_ConstraintRule() for RELAX_EXP type."""
        link_expr = LinkExpr(
            type=LinkType.RELAX_EXP,
            target=self.pref_s0_p0_amp,
            driver=self.pref_s0_p1_amp,
            args={"T": 1.0, "A": 1.5, "C": 0.2, "t_override": 0.5},
            enabled=False
        )
        
        rule = LinkExpr_to_ConstraintRule(link_expr)
        
        self.assertIsInstance(rule, RelaxDecayConstraint)
        self.assertEqual(rule.target_pref, self.pref_s0_p0_amp)
        self.assertEqual(rule.driver_pref, self.pref_s0_p1_amp)
        self.assertAlmostEqual(rule.T_seconds, 1.0)
        self.assertAlmostEqual(rule.A, 1.5)
        self.assertAlmostEqual(rule.C, 0.2)
        self.assertAlmostEqual(rule.time_seconds, 0.5)
        self.assertEqual(rule.enabled, False)

    def test_migration_constraint_rule_to_link_expr_linear(self):
        """Test ConstraintRule_to_LinkExpr() for LinearConstraint."""
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.5,
            b=0.5,
            enabled=True
        )
        
        link_expr = ConstraintRule_to_LinkExpr(rule)
        
        self.assertEqual(link_expr.type, LinkType.LINEAR)
        self.assertEqual(link_expr.target, self.pref_s0_p0_amp)
        self.assertEqual(link_expr.driver, self.pref_s0_p1_amp)
        self.assertAlmostEqual(link_expr.args["a"], 2.5)
        self.assertAlmostEqual(link_expr.args["b"], 0.5)
        self.assertEqual(link_expr.enabled, True)

    def test_migration_constraint_rule_to_link_expr_relax_decay(self):
        """Test ConstraintRule_to_LinkExpr() for RelaxDecayConstraint."""
        rule = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=1.0,
            A=1.5,
            C=0.2,
            time_seconds=0.5,
            T_name=None,
            enabled=True
        )
        
        link_expr = ConstraintRule_to_LinkExpr(rule)
        
        self.assertEqual(link_expr.type, LinkType.RELAX_EXP)
        self.assertEqual(link_expr.target, self.pref_s0_p0_amp)
        self.assertEqual(link_expr.driver, self.pref_s0_p1_amp)
        self.assertAlmostEqual(link_expr.args["T"], 1.0)
        self.assertAlmostEqual(link_expr.args["A"], 1.5)
        self.assertAlmostEqual(link_expr.args["C"], 0.2)
        self.assertAlmostEqual(link_expr.args["t_override"], 0.5)
        self.assertEqual(link_expr.enabled, True)

    def test_migration_round_trip_linear(self):
        """Test full round-trip: LinearConstraint → LinkExpr → LinearConstraint."""
        original = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=3.0,
            b=1.5,
            enabled=False
        )
        
        # Round trip
        link_expr = ConstraintRule_to_LinkExpr(original)
        restored = LinkExpr_to_ConstraintRule(link_expr)
        
        # Verify
        self.assertIsInstance(restored, LinearConstraint)
        self.assertEqual(restored.target_pref, original.target_pref)
        self.assertEqual(restored.driver_pref, original.driver_pref)
        self.assertAlmostEqual(restored.a, original.a)
        self.assertAlmostEqual(restored.b, original.b)
        self.assertEqual(restored.enabled, original.enabled)

    def test_migration_round_trip_relax_decay(self):
        """Test full round-trip: RelaxDecayConstraint → LinkExpr → RelaxDecayConstraint."""
        original = RelaxDecayConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            T_seconds=0.8,
            A=2.0,
            C=0.1,
            time_seconds=0.2,
            T_name=None,
            enabled=True
        )
        
        # Round trip
        link_expr = ConstraintRule_to_LinkExpr(original)
        restored = LinkExpr_to_ConstraintRule(link_expr)
        
        # Verify
        self.assertIsInstance(restored, RelaxDecayConstraint)
        self.assertEqual(restored.target_pref, original.target_pref)
        self.assertEqual(restored.driver_pref, original.driver_pref)
        self.assertAlmostEqual(restored.T_seconds, original.T_seconds)
        self.assertAlmostEqual(restored.A, original.A)
        self.assertAlmostEqual(restored.C, original.C)
        self.assertAlmostEqual(restored.time_seconds, original.time_seconds)
        self.assertEqual(restored.enabled, original.enabled)

    # ========================================================================
    # ConstrainedPeak Tests
    # ========================================================================

    def test_constrained_peak_creation(self):
        """Test basic ConstrainedPeak creation."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0
        )
        
        cp = ConstrainedPeak(peak=peak, constraints={"amp": constraint})
        
        self.assertEqual(cp.peak.pos, 100.0)
        self.assertEqual(cp.peak.amp, 50.0)
        self.assertIn("amp", cp.constraints)
        self.assertIsInstance(cp.constraints["amp"], LinearConstraint)

    def test_constrained_peak_empty_constraints(self):
        """Test ConstrainedPeak with no constraints."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        cp = ConstrainedPeak(peak=peak)
        
        self.assertEqual(len(cp.constraints), 0)
        self.assertIsNone(cp.get_constraint("amp"))

    def test_constrained_peak_get_constraint(self):
        """Test get_constraint() method."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0
        )
        
        cp = ConstrainedPeak(peak=peak, constraints={"amp": constraint})
        
        retrieved = cp.get_constraint("amp")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.a, 2.0)
        
        # Case insensitive
        retrieved_upper = cp.get_constraint("AMP")
        self.assertIsNotNone(retrieved_upper)

    def test_constrained_peak_set_constraint(self):
        """Test set_constraint() method."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        cp = ConstrainedPeak(peak=peak)
        
        # Add constraint
        constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0
        )
        cp.set_constraint("amp", constraint)
        self.assertIsNotNone(cp.get_constraint("amp"))
        
        # Remove constraint
        cp.set_constraint("amp", None)
        self.assertIsNone(cp.get_constraint("amp"))

    def test_constrained_peak_validate_success(self):
        """Test validate() with valid constraints."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        
        cp = ConstrainedPeak(peak=peak, constraints={"amp": constraint})
        
        # Build driver peaks dict
        driver_peaks = {
            (0, 1): self.peak_s0_p1
        }
        
        errors = cp.validate(
            target_pref=self.pref_s0_p0_amp,
            driver_peaks=driver_peaks,
            slice_states=self.slice_states
        )
        
        self.assertEqual(len(errors), 0)

    def test_constrained_peak_validate_missing_driver(self):
        """Test validate() detects missing driver."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        
        cp = ConstrainedPeak(peak=peak, constraints={"amp": constraint})
        
        # Empty driver peaks dict
        driver_peaks = {}
        
        errors = cp.validate(
            target_pref=self.pref_s0_p0_amp,
            driver_peaks=driver_peaks,
            slice_states=self.slice_states
        )
        
        self.assertTrue(len(errors) > 0)

    def test_constrained_peak_validate_disabled_constraint(self):
        """Test validate() skips disabled constraints."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=False  # Disabled
        )
        
        cp = ConstrainedPeak(peak=peak, constraints={"amp": constraint})
        
        # Even though driver doesn't exist, disabled constraint should not error
        driver_peaks = {}
        
        errors = cp.validate(
            target_pref=self.pref_s0_p0_amp,
            driver_peaks=driver_peaks,
            slice_states=self.slice_states
        )
        
        self.assertEqual(len(errors), 0)

    def test_constrained_peak_multiple_constraints(self):
        """Test ConstrainedPeak with multiple constraints."""
        from constraint_rules import ConstrainedPeak
        
        peak = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        
        amp_constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        
        pos_constraint = LinearConstraint(
            target_pref=self.pref_s0_p0_pos,
            driver_pref=self.pref_s0_p1_pos,
            a=1.0,
            b=5.0,
            enabled=True
        )
        
        cp = ConstrainedPeak(
            peak=peak,
            constraints={"amp": amp_constraint, "pos": pos_constraint}
        )
        
        self.assertEqual(len(cp.constraints), 2)
        self.assertIsNotNone(cp.get_constraint("amp"))
        self.assertIsNotNone(cp.get_constraint("pos"))

    # ========== Phase 3: ConstraintStore Registry Tests ==========
    
    def test_constraint_store_creation(self):
        """Test ConstraintStore initialization."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        self.assertEqual(len(store.constraints), 0)
        self.assertEqual(len(store.reverse_index), 0)
    
    def test_constraint_store_add_constraint(self):
        """Test adding constraints to ConstraintStore."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        self.assertEqual(len(store.constraints), 1)
        retrieved = store.get_constraint(self.pref_s0_p0_amp)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.a, 2.0)
    
    def test_constraint_store_remove_constraint(self):
        """Test removing constraints from ConstraintStore."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        
        store.add_constraint(self.pref_s0_p0_amp, rule)
        self.assertEqual(len(store.constraints), 1)
        
        store.add_constraint(self.pref_s0_p0_amp, None)
        self.assertEqual(len(store.constraints), 0)
    
    def test_constraint_store_reverse_index(self):
        """Test reverse index for dependent tracking."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        # Add constraint: p0_amp depends on p1_amp
        rule1 = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule1)
        
        # Check reverse index
        dependents = store.get_dependents(self.pref_s0_p1_amp)
        self.assertEqual(len(dependents), 1)
        self.assertIn(self.pref_s0_p0_amp, dependents)
    
    def test_constraint_store_multiple_dependents(self):
        """Test reverse index with multiple dependents on same driver."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        # p0_amp depends on p1_amp
        rule1 = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule1)
        
        # p0_pos also depends on p1_amp (same driver)
        rule2 = LinearConstraint(
            target_pref=self.pref_s0_p0_pos,
            driver_pref=self.pref_s0_p1_amp,
            a=1.5,
            b=0.5,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_pos, rule2)
        
        # Check reverse index
        dependents = store.get_dependents(self.pref_s0_p1_amp)
        self.assertEqual(len(dependents), 2)
        self.assertIn(self.pref_s0_p0_amp, dependents)
        self.assertIn(self.pref_s0_p0_pos, dependents)
    
    def test_constraint_store_validate_all_success(self):
        """Test validate_all with valid constraints."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        # Create peak map
        peak_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        peak_p1 = Peak(pos=105.0, amp=60.0, lor_hz=1.2, gauss_disp=0.6)
        peak_map = {
            (0, 0): peak_p0,
            (0, 1): peak_p1
        }
        
        # Validate
        errors = store.validate_all(peak_map, self.slice_states)
        self.assertEqual(len(errors), 0)
    
    def test_constraint_store_validate_all_missing_driver(self):
        """Test validate_all with missing driver peak."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        # Only target peak, no driver
        peak_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        peak_map = {
            (0, 0): peak_p0
        }
        
        # Validate
        errors = store.validate_all(peak_map, self.slice_states)
        self.assertEqual(len(errors), 1)
        self.assertIn("Driver peak not found", errors[0].message)
    
    def test_constraint_store_validate_all_disabled(self):
        """Test validate_all skips disabled constraints."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=False  # Disabled
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        # Only target peak, no driver
        peak_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        peak_map = {
            (0, 0): peak_p0
        }
        
        # Validate - should skip disabled constraint
        errors = store.validate_all(peak_map, self.slice_states)
        self.assertEqual(len(errors), 0)
    
    def test_constraint_store_apply_to_lmfit(self):
        """Test apply_to_lmfit with ConstraintStore."""
        from constraint_rules import ConstraintStore
        from lmfit import Parameters
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        # Create peak map
        peak_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        peak_p1 = Peak(pos=105.0, amp=60.0, lor_hz=1.2, gauss_disp=0.6)
        peak_map = {
            (0, 0): peak_p0,
            (0, 1): peak_p1
        }
        
        # Create lmfit parameters
        params = Parameters()
        params.add('amp_0', value=50.0, vary=True)
        params.add('amp_1', value=60.0, vary=True)
        params.add('pos_0', value=100.0, vary=True)
        params.add('pos_1', value=105.0, vary=True)
        
        # Apply constraints
        store.apply_to_lmfit(params, peak_map, self.slice_states)
        
        # Check that amp_0 has expression
        p_amp_0 = params.get('amp_0')
        self.assertIsNotNone(p_amp_0)
        self.assertIsNotNone(p_amp_0.expr)
        self.assertIn('amp_1', p_amp_0.expr)
    
    def test_constraint_store_serialization(self):
        """Test ConstraintStore to_dict and from_dict."""
        from constraint_rules import ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        # Serialize
        data = store.to_dict()
        self.assertIn('constraints', data)
        self.assertGreater(len(data['constraints']), 0)
        
        # Deserialize
        store2 = ConstraintStore.from_dict(data)
        self.assertEqual(len(store2.constraints), 1)
        
        # Check deserialized rule
        retrieved = store2.get_constraint(self.pref_s0_p0_amp)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.a, 2.0)
        self.assertEqual(retrieved.b, 1.0)

    # ========== Phase 5: FitOrchestrator Tests ==========
    
    def test_fit_orchestrator_creation(self):
        """Test FitOrchestrator initialization."""
        from constraint_rules import FitOrchestrator
        
        orchestrator = FitOrchestrator()
        self.assertIsNone(orchestrator.constraint_store)
    
    def test_fit_orchestrator_with_constraint_store(self):
        """Test FitOrchestrator with ConstraintStore."""
        from constraint_rules import FitOrchestrator, ConstraintStore
        
        store = ConstraintStore()
        orchestrator = FitOrchestrator(constraint_store=store)
        
        self.assertIsNotNone(orchestrator.constraint_store)
        self.assertIsInstance(orchestrator.constraint_store, ConstraintStore)
    
    def test_fit_orchestrator_validate_constraints_empty(self):
        """Test FitOrchestrator.validate_constraints with empty store."""
        from constraint_rules import FitOrchestrator
        
        orchestrator = FitOrchestrator()
        errors = orchestrator.validate_constraints({}, {})
        
        self.assertEqual(len(errors), 0)
    
    def test_fit_orchestrator_validate_constraints_valid(self):
        """Test FitOrchestrator.validate_constraints with valid constraints."""
        from constraint_rules import FitOrchestrator, ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        orchestrator = FitOrchestrator(constraint_store=store)
        
        # Create peak map
        peak_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        peak_p1 = Peak(pos=105.0, amp=60.0, lor_hz=1.2, gauss_disp=0.6)
        peak_map = {
            (0, 0): peak_p0,
            (0, 1): peak_p1
        }
        
        # Validate
        errors = orchestrator.validate_constraints(peak_map, self.slice_states)
        self.assertEqual(len(errors), 0)
    
    def test_fit_orchestrator_validate_constraints_invalid(self):
        """Test FitOrchestrator.validate_constraints with invalid constraints."""
        from constraint_rules import FitOrchestrator, ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        orchestrator = FitOrchestrator(constraint_store=store)
        
        # Only target peak, no driver - should generate error
        peak_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        peak_map = {
            (0, 0): peak_p0
        }
        
        # Validate
        errors = orchestrator.validate_constraints(peak_map, self.slice_states)
        self.assertGreater(len(errors), 0)
        self.assertIn("Driver peak not found", errors[0].message)
    
    def test_fit_orchestrator_apply_constraints(self):
        """Test FitOrchestrator.apply_constraints_to_lmfit."""
        from constraint_rules import FitOrchestrator, ConstraintStore
        from lmfit import Parameters
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        orchestrator = FitOrchestrator(constraint_store=store)
        
        # Create peak map
        peak_p0 = Peak(pos=100.0, amp=50.0, lor_hz=1.0, gauss_disp=0.5)
        peak_p1 = Peak(pos=105.0, amp=60.0, lor_hz=1.2, gauss_disp=0.6)
        peak_map = {
            (0, 0): peak_p0,
            (0, 1): peak_p1
        }
        
        # Create lmfit parameters
        params = Parameters()
        params.add('amp_0', value=50.0, vary=True)
        params.add('amp_1', value=60.0, vary=True)
        params.add('pos_0', value=100.0, vary=True)
        params.add('pos_1', value=105.0, vary=True)
        
        # Apply constraints via orchestrator
        orchestrator.apply_constraints_to_lmfit(params, peak_map, self.slice_states)
        
        # Check that amp_0 has expression
        p_amp_0 = params.get('amp_0')
        self.assertIsNotNone(p_amp_0)
        self.assertIsNotNone(p_amp_0.expr)
        self.assertIn('amp_1', p_amp_0.expr)
    
    def test_fit_orchestrator_from_constraint_store(self):
        """Test FitOrchestrator.from_constraint_store static method."""
        from constraint_rules import FitOrchestrator, ConstraintStore
        
        store = ConstraintStore()
        
        rule = LinearConstraint(
            target_pref=self.pref_s0_p0_amp,
            driver_pref=self.pref_s0_p1_amp,
            a=2.0,
            b=1.0,
            enabled=True
        )
        store.add_constraint(self.pref_s0_p0_amp, rule)
        
        # Create via static method
        orchestrator = FitOrchestrator.from_constraint_store(store)
        
        self.assertIsNotNone(orchestrator.constraint_store)
        self.assertEqual(len(orchestrator.constraint_store.constraints), 1)
    
    def test_fit_orchestrator_from_link_store(self):
        """Test FitOrchestrator.from_link_store static method."""
        from constraint_rules import FitOrchestrator
        from nmrFit_v0 import LinkStore, LinkExpr, LinkType
        
        # Create a legacy LinkStore
        link_store = LinkStore()
        
        expr = LinkExpr(
            type=LinkType.LINEAR,
            target=self.pref_s0_p0_amp,
            driver=self.pref_s0_p1_amp,
            args={'a': 2.0, 'b': 1.0},
            enabled=True
        )
        link_store.set_link(expr)
        
        # Create orchestrator via from_link_store
        orchestrator = FitOrchestrator.from_link_store(link_store)
        
        self.assertIsNotNone(orchestrator.constraint_store)
        self.assertEqual(len(orchestrator.constraint_store.constraints), 1)
    
    def test_fit_orchestrator_from_link_store_none(self):
        """Test FitOrchestrator.from_link_store with None."""
        from constraint_rules import FitOrchestrator
        
        orchestrator = FitOrchestrator.from_link_store(None)
        
        self.assertIsNone(orchestrator.constraint_store)
    
    def test_fit_orchestrator_repr(self):
        """Test FitOrchestrator string representation."""
        from constraint_rules import FitOrchestrator, ConstraintStore
        
        # Empty orchestrator
        orchestrator1 = FitOrchestrator()
        repr1 = repr(orchestrator1)
        self.assertIn("None", repr1)
        
        # With constraint store
        orchestrator2 = FitOrchestrator(constraint_store=ConstraintStore())
        repr2 = repr(orchestrator2)
        self.assertIn("ConstraintStore", repr2)


if __name__ == "__main__":
    unittest.main()

