"""
Solution checker to detect existing simulation results and avoid recomputation.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class SolutionChecker:
    """
    Check for existing simulation results and determine what needs to be recomputed.
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialize solution checker.

        Args:
            results_dir: Base directory for results
        """
        self.results_dir = results_dir

    def check_standard_comparison(self, reynolds_numbers: List[int]) -> Dict:
        """
        Check for existing standard comparison results.

        Args:
            reynolds_numbers: List of Reynolds numbers to check

        Returns:
            Dictionary with check results
        """
        results = {
            'has_results': False,
            'missing_re': [],
            'existing_re': [],
            'needs_computation': False,
            'existing_files': []
        }

        # Check for comparison results
        comparison_dir = os.path.join(self.results_dir, "comparison")
        if os.path.exists(comparison_dir):
            # Check for timing summary
            timing_file = os.path.join(comparison_dir, "timing_summary.json")
            if os.path.exists(timing_file):
                try:
                    with open(timing_file, 'r') as f:
                        timing_data = json.load(f)

                    # Check if all Reynolds numbers are present and have valid data
                    existing_re = []
                    for re in reynolds_numbers:
                        if str(re) in timing_data.get('timing_comparison', {}):
                            timing_info = timing_data['timing_comparison'][str(re)]
                            # Check if the timing data has both FEM and LBM results
                            if 'fem_time' in timing_info and 'lbm_time' in timing_info:
                                if timing_info['fem_time'] > 0 and timing_info['lbm_time'] > 0:
                                    existing_re.append(re)
                                else:
                                    results['missing_re'].append(re)
                            else:
                                results['missing_re'].append(re)
                        else:
                            results['missing_re'].append(re)

                    results['existing_re'] = existing_re
                    results['has_results'] = len(existing_re) > 0
                    results['needs_computation'] = len(results['missing_re']) > 0

                except (json.JSONDecodeError, KeyError):
                    results['needs_computation'] = True
            else:
                results['needs_computation'] = True
        else:
            results['needs_computation'] = True

        # Check for individual FEM and LBM results
        for re in reynolds_numbers:
            fem_file = os.path.join(self.results_dir, "fem", f"fem_Re{re}_results.npz")
            lbm_file = os.path.join(self.results_dir, "lbm", f"lbm_Re{re}_results.npz")

            if os.path.exists(fem_file) and os.path.exists(lbm_file):
                results['existing_files'].append(f"Re{re}")
            else:
                if re not in results['missing_re']:
                    results['missing_re'].append(re)

        return results

    def check_initial_condition_comparison(self, conditions: List[str], reynolds_numbers: List[int]) -> Dict:
        """
        Check for existing initial condition comparison results.

        Args:
            conditions: List of initial condition types
            reynolds_numbers: List of Reynolds numbers to check

        Returns:
            Dictionary with check results
        """
        results = {
            'has_results': False,
            'missing_conditions': [],
            'existing_conditions': [],
            'needs_computation': False,
            'existing_files': []
        }

        # Check for initial condition comparison results
        ic_dir = os.path.join(self.results_dir, "initial_condition_comparison")
        if os.path.exists(ic_dir):
            # Check for summary data
            summary_file = os.path.join(ic_dir, "data", "summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)

                    # Check if all conditions are present and have valid data
                    for condition in conditions:
                        if condition in summary_data and summary_data[condition]:
                            # Check if the condition has both FEM and LBM results
                            condition_data = summary_data[condition]
                            if 'fem' in condition_data and 'lbm' in condition_data:
                                if condition_data['fem'] and condition_data['lbm']:
                                    results['existing_conditions'].append(condition)
                                else:
                                    results['missing_conditions'].append(condition)
                            else:
                                results['missing_conditions'].append(condition)
                        else:
                            results['missing_conditions'].append(condition)

                    results['has_results'] = len(results['existing_conditions']) > 0
                    results['needs_computation'] = len(results['missing_conditions']) > 0

                except (json.JSONDecodeError, KeyError):
                    results['needs_computation'] = True
            else:
                results['needs_computation'] = True
        else:
            results['needs_computation'] = True

        # Check for individual condition results
        for condition in conditions:
            for re in reynolds_numbers:
                fem_file = os.path.join(self.results_dir, "fem", f"fem_Re{re}_results.npz")
                lbm_file = os.path.join(self.results_dir, "lbm", f"lbm_Re{re}_results.npz")

                if os.path.exists(fem_file) and os.path.exists(lbm_file):
                    results['existing_files'].append(f"{condition}_Re{re}")
                else:
                    if condition not in results['missing_conditions']:
                        results['missing_conditions'].append(condition)

        return results

    def check_animations(self, animation_types: List[str], reynolds_numbers: List[int]) -> Dict:
        """
        Check for existing animations.

        Args:
            animation_types: List of animation types (pressure, velocity, vorticity)
            reynolds_numbers: List of Reynolds numbers to check

        Returns:
            Dictionary with check results
        """
        results = {
            'has_animations': False,
            'missing_animations': [],
            'existing_animations': [],
            'needs_generation': False
        }

        # Check standard animations
        anim_dir = os.path.join(self.results_dir, "animations")
        if os.path.exists(anim_dir):
            for re in reynolds_numbers:
                for anim_type in animation_types:
                    fem_file = os.path.join(anim_dir, f"fem_{anim_type}_Re{re}_animation.gif")
                    lbm_file = os.path.join(anim_dir, f"lbm_{anim_type}_Re{re}_animation.gif")
                    comp_file = os.path.join(anim_dir, f"comparison_{anim_type}_Re{re}_animation.gif")

                    if all(os.path.exists(f) for f in [fem_file, lbm_file, comp_file]):
                        results['existing_animations'].append(f"{anim_type}_Re{re}")
                    else:
                        results['missing_animations'].append(f"{anim_type}_Re{re}")

        results['has_animations'] = len(results['existing_animations']) > 0
        results['needs_generation'] = len(results['missing_animations']) > 0

        return results

    def check_initial_condition_animations(self, conditions: List[str], animation_types: List[str]) -> Dict:
        """
        Check for existing initial condition animations.

        Args:
            conditions: List of initial condition types
            animation_types: List of animation types

        Returns:
            Dictionary with check results
        """
        results = {
            'has_animations': False,
            'missing_animations': [],
            'existing_animations': [],
            'needs_generation': False
        }

        # Check initial condition animations
        ic_anim_dir = os.path.join(self.results_dir, "animations", "initial_conditions")
        if os.path.exists(ic_anim_dir):
            for condition in conditions:
                for anim_type in animation_types:
                    fem_file = os.path.join(ic_anim_dir, f"fem_{anim_type}_{condition}_Re{100 if condition != 'steady' else 20}_animation.gif")
                    lbm_file = os.path.join(ic_anim_dir, f"lbm_{anim_type}_{condition}_Re{100 if condition != 'steady' else 20}_animation.gif")
                    comp_file = os.path.join(ic_anim_dir, f"comparison_{anim_type}_{condition}_Re{100 if condition != 'steady' else 20}_animation.gif")

                    if all(os.path.exists(f) for f in [fem_file, lbm_file, comp_file]):
                        results['existing_animations'].append(f"{anim_type}_{condition}")
                    else:
                        results['missing_animations'].append(f"{anim_type}_{condition}")

        results['has_animations'] = len(results['existing_animations']) > 0
        results['needs_generation'] = len(results['missing_animations']) > 0

        return results

    def get_computation_plan(self, mode: str = "all") -> Dict:
        """
        Get a computation plan based on existing results.

        Args:
            mode: Computation mode ("all", "standard", "initial_conditions", "animations", "ic_animations")

        Returns:
            Dictionary with computation plan
        """
        plan = {
            'run_standard_comparison': False,
            'run_initial_condition_comparison': False,
            'run_standard_animations': False,
            'run_ic_animations': False,
            'run_test_ic': False,
            'reasons': []
        }

        if mode in ["all", "standard"]:
            # Check standard comparison
            standard_check = self.check_standard_comparison([20, 40, 100, 200])
            if standard_check['needs_computation']:
                plan['run_standard_comparison'] = True
                plan['reasons'].append(f"Missing standard comparison for Re: {standard_check['missing_re']}")
            else:
                plan['reasons'].append("Standard comparison results found, skipping computation")

        if mode in ["all", "initial_conditions"]:
            # Check initial condition comparison
            ic_check = self.check_initial_condition_comparison(["steady", "unsteady", "oscillating"], [20, 100])
            if ic_check['needs_computation']:
                plan['run_initial_condition_comparison'] = True
                plan['reasons'].append(f"Missing initial condition comparison for: {ic_check['missing_conditions']}")
            else:
                plan['reasons'].append("Initial condition comparison results found, skipping computation")

        if mode in ["all", "animations"]:
            # Check standard animations
            anim_check = self.check_animations(["pressure", "velocity", "vorticity"], [20, 40, 100, 200])
            if anim_check['needs_generation']:
                plan['run_standard_animations'] = True
                plan['reasons'].append(f"Missing standard animations: {anim_check['missing_animations']}")
            else:
                plan['reasons'].append("Standard animations found, skipping generation")

        if mode in ["all", "ic_animations"]:
            # Check initial condition animations
            ic_anim_check = self.check_initial_condition_animations(["steady", "unsteady", "oscillating"], ["pressure", "velocity", "vorticity"])
            if ic_anim_check['needs_generation']:
                plan['run_ic_animations'] = True
                plan['reasons'].append(f"Missing initial condition animations: {ic_anim_check['missing_animations']}")
            else:
                plan['reasons'].append("Initial condition animations found, skipping generation")

        if mode in ["all", "test_ic"]:
            # Always run test_ic as it's quick validation
            plan['run_test_ic'] = True
            plan['reasons'].append("Running initial condition validation")

        return plan

    def print_status(self, mode: str = "all"):
        """
        Print status of existing results.

        Args:
            mode: Computation mode
        """
        print("Solution Checker - Existing Results Status")
        print("=" * 50)

        plan = self.get_computation_plan(mode)

        print(f"Mode: {mode}")
        print(f"Computation Plan:")
        print(f"  Standard comparison: {'YES' if plan['run_standard_comparison'] else 'NO'}")
        print(f"  Initial condition comparison: {'YES' if plan['run_initial_condition_comparison'] else 'NO'}")
        print(f"  Standard animations: {'YES' if plan['run_standard_animations'] else 'NO'}")
        print(f"  Initial condition animations: {'YES' if plan['run_ic_animations'] else 'NO'}")
        print(f"  Test initial conditions: {'YES' if plan['run_test_ic'] else 'NO'}")

        print(f"\nReasons:")
        for reason in plan['reasons']:
            print(f"  - {reason}")

        return plan


def main():
    """Test the solution checker."""
    checker = SolutionChecker()

    print("Testing Solution Checker")
    print("=" * 30)

    # Check all modes
    for mode in ["all", "standard", "initial_conditions", "animations", "ic_animations"]:
        print(f"\nMode: {mode}")
        plan = checker.get_computation_plan(mode)
        print(f"  Run standard: {plan['run_standard_comparison']}")
        print(f"  Run IC comparison: {plan['run_initial_condition_comparison']}")
        print(f"  Run standard animations: {plan['run_standard_animations']}")
        print(f"  Run IC animations: {plan['run_ic_animations']}")
        print(f"  Run test IC: {plan['run_test_ic']}")


if __name__ == "__main__":
    main()
