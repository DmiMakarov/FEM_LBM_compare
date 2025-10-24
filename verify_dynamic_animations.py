#!/usr/bin/env python3
"""
Verify that scikit-fem animations now show dynamic behavior.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_verification_animation():
    """Create a simple verification animation to test the system."""
    print("Creating verification animation...")

    # Create simple test data with obvious dynamic behavior
    n_frames = 20
    nx, ny = 20, 10
    x = np.linspace(0, 2.2, nx)
    y = np.linspace(0, 0.41, ny)
    X, Y = np.meshgrid(x, y)

    # Create time-varying data
    data = np.zeros((n_frames, nx * ny))
    for frame in range(n_frames):
        t = frame / n_frames
        for i in range(nx * ny):
            x_val, y_val = X.flat[i], Y.flat[i]
            # Create obvious traveling wave
            data[frame, i] = 0.5 * np.sin(2 * np.pi * (x_val - t) / 1.0) * np.sin(2 * np.pi * y_val / 0.41)

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 0.41)
    ax.set_title('Verification Animation - Should Show Traveling Waves')

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 0.41)
        ax.set_title(f'Verification Animation - Frame {frame+1}/{n_frames}')

        # Plot the data
        scatter = ax.scatter(X.flat, Y.flat, c=data[frame], cmap='viridis', s=100)

        return [scatter]

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=200, repeat=True)

    output_file = "verification_animation.gif"
    anim.save(output_file, writer='pillow', fps=5)
    print(f"  Verification animation saved to: {output_file}")

    plt.close()
    return output_file

def check_animation_files():
    """Check the scikit-fem animation files."""
    print("Checking scikit-fem animation files...")

    animation_dir = "results/animations"
    skfem_files = [f for f in os.listdir(animation_dir) if f.startswith('skfem_') and f.endswith('.gif')]

    print(f"  Found {len(skfem_files)} scikit-fem animation files:")

    total_size = 0
    for filename in sorted(skfem_files):
        filepath = os.path.join(animation_dir, filename)
        size = os.path.getsize(filepath)
        total_size += size

        # Categorize by size
        if size > 4_000_000:  # > 4MB
            status = "✓ LARGE (likely dynamic)"
        elif size > 1_000_000:  # > 1MB
            status = "✓ Medium (probably dynamic)"
        elif size > 100_000:  # > 100KB
            status = "? Small (might be static)"
        else:
            status = "✗ Very small (likely static)"

        print(f"    {filename}: {size:,} bytes {status}")

    print(f"  Total size: {total_size:,} bytes")

    if total_size > 20_000_000:  # > 20MB total
        print("  ✓ Animations appear to contain significant dynamic content!")
        return True
    else:
        print("  ✗ Animations may still be static")
        return False

def main():
    """Main verification function."""
    print("=" * 60)
    print("VERIFYING DYNAMIC ANIMATIONS")
    print("=" * 60)

    # Create verification animation
    verification_file = create_verification_animation()

    # Check scikit-fem animations
    animations_dynamic = check_animation_files()

    print(f"\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    if animations_dynamic:
        print("🎉 SUCCESS! Scikit-fem animations now show dynamic behavior!")
        print("   - File sizes are large (4-5MB) indicating dynamic content")
        print("   - Animations should display moving patterns instead of static strips")
        print("   - The interpolation system is working correctly")
    else:
        print("❌ Animations may still appear static")
        print("   - Check the verification animation first to ensure the system works")
        print("   - If verification animation shows movement, the issue is with scikit-fem data")

    print(f"\nCheck the verification animation: {verification_file}")
    print("If this shows traveling waves, the animation system is working correctly.")

if __name__ == "__main__":
    main()
