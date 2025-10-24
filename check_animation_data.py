#!/usr/bin/env python3
"""
Check if scikit-fem animation data contains dynamic behavior.
"""

import numpy as np
import os
import sys

def check_skfem_data():
    """Check if scikit-fem data contains dynamic behavior."""
    print("Checking scikit-fem data for dynamic behavior...")

    # Look for scikit-fem result files
    result_files = []
    for filename in os.listdir('results/skfem'):
        if filename.startswith('skfem_') and filename.endswith('.npz'):
            result_files.append(os.path.join('results/skfem', filename))

    if not result_files:
        print("  No scikit-fem result files found")
        return False

    print(f"  Found {len(result_files)} scikit-fem result files")

    for result_file in result_files:
        print(f"\n  Analyzing: {os.path.basename(result_file)}")

        try:
            data = np.load(result_file)

            # Check velocity data
            if 'velocity_x_fields' in data and 'velocity_y_fields' in data:
                ux_data = data['velocity_x_fields']
                uy_data = data['velocity_y_fields']

                print(f"    Velocity data shape: {ux_data.shape}")

                # Check for variation over time
                ux_std = np.std(ux_data, axis=0)
                uy_std = np.std(uy_data, axis=0)

                max_ux_std = np.max(ux_std)
                max_uy_std = np.max(uy_std)

                print(f"    Max ux std over time: {max_ux_std:.6f}")
                print(f"    Max uy std over time: {max_uy_std:.6f}")

                if max_ux_std > 0.001 or max_uy_std > 0.001:
                    print("    ✓ Velocity shows dynamic behavior")
                else:
                    print("    ✗ Velocity appears static")

            # Check pressure data
            if 'pressure_fields' in data:
                p_data = data['pressure_fields']
                print(f"    Pressure data shape: {p_data.shape}")

                p_std = np.std(p_data, axis=0)
                max_p_std = np.max(p_std)

                print(f"    Max pressure std over time: {max_p_std:.6f}")

                if max_p_std > 0.001:
                    print("    ✓ Pressure shows dynamic behavior")
                else:
                    print("    ✗ Pressure appears static")

            # Check vorticity data
            if 'vorticity_fields' in data:
                vort_data = data['vorticity_fields']
                print(f"    Vorticity data shape: {vort_data.shape}")

                vort_std = np.std(vort_data, axis=0)
                max_vort_std = np.max(vort_std)

                print(f"    Max vorticity std over time: {max_vort_std:.6f}")

                if max_vort_std > 0.001:
                    print("    ✓ Vorticity shows dynamic behavior")
                else:
                    print("    ✗ Vorticity appears static")

            # Check time data
            if 'time' in data:
                time_data = data['time']
                print(f"    Time data: {len(time_data)} steps, range: {time_data[0]:.3f} to {time_data[-1]:.3f}")

        except Exception as e:
            print(f"    Error analyzing {result_file}: {e}")

    return True

def main():
    """Main function."""
    print("=" * 60)
    print("CHECKING SCIKIT-FEM ANIMATION DATA")
    print("=" * 60)

    check_skfem_data()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("If the data shows dynamic behavior (std > 0.001),")
    print("the animations should display moving patterns instead of static strips.")

if __name__ == "__main__":
    main()
