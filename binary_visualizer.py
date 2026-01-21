"""
Binary Operations Visualizer for BNN Research
Demonstrates XNOR and popcount operations used in Binary Neural Networks
Author: Vakkalagadda Tanush Pavan
"""

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class BinaryOpsVisualizer:
    """Visualize and compare binary operations vs traditional MAC"""
    
    def __init__(self):
        self.results = {}
    
    def traditional_mac(self, weights: np.ndarray, activations: np.ndarray) -> float:
        """Traditional Multiply-Accumulate operation"""
        return np.sum(weights * activations)
    
    def xnor_popcount(self, binary_weights: np.ndarray, binary_activations: np.ndarray) -> int:
        """
        XNOR-popcount operation for BNN
        XNOR gives 1 when bits match, 0 when different
        Popcount sums the matching bits
        """
        xnor_result = np.logical_not(np.logical_xor(binary_weights, binary_activations))
        popcount = np.sum(xnor_result)
        return popcount
    
    def binarize(self, values: np.ndarray) -> np.ndarray:
        """Convert float values to binary {0, 1}"""
        return (values > 0).astype(int)
    
    def compare_operations(self, size: int = 8) -> dict:
        """Compare MAC vs XNOR-popcount for random inputs"""
        # Generate random weights and activations
        weights = np.random.randn(size)
        activations = np.random.randn(size)
        
        # Traditional MAC
        mac_result = self.traditional_mac(weights, activations)
        
        # Binarize and compute XNOR-popcount
        binary_weights = self.binarize(weights)
        binary_activations = self.binarize(activations)
        xnor_result = self.xnor_popcount(binary_weights, binary_activations)
        
        # Approximate MAC result from popcount (scaling)
        approximate_mac = 2 * xnor_result - size
        
        results = {
            'weights': weights,
            'activations': activations,
            'mac_result': mac_result,
            'xnor_popcount': xnor_result,
            'approximate_mac': approximate_mac,
            'binary_weights': binary_weights,
            'binary_activations': binary_activations
        }
        
        self.results = results
        return results
    
    def calculate_efficiency(self, input_size: int) -> dict:
        """Calculate power and area efficiency gains"""
        # Approximate metrics based on research
        mac_power = input_size * 4.6  # pJ per MAC operation
        xnor_power = input_size * 0.08  # pJ per XNOR-popcount
        
        mac_area = input_size * 282  # µm² per MAC
        xnor_area = input_size * 18  # µm² per XNOR
        
        power_reduction = (1 - xnor_power / mac_power) * 100
        area_reduction = (1 - xnor_area / mac_area) * 100
        
        return {
            'power_reduction_percent': power_reduction,
            'area_reduction_percent': area_reduction,
            'speedup': mac_power / xnor_power
        }
    
    def visualize_comparison(self):
        """Create visualization of the operations"""
        if not self.results:
            print("Run compare_operations() first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Input weights
        axes[0, 0].stem(self.results['weights'])
        axes[0, 0].set_title('Original Weights (Float)')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Binary weights
        axes[0, 1].stem(self.results['binary_weights'])
        axes[0, 1].set_title('Binarized Weights {0, 1}')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Binary Value')
        axes[0, 1].set_ylim([-0.5, 1.5])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: XNOR operation
        xnor_result = np.logical_not(
            np.logical_xor(self.results['binary_weights'], 
                          self.results['binary_activations'])
        ).astype(int)
        axes[1, 0].stem(xnor_result)
        axes[1, 0].set_title('XNOR Result (1 = Match, 0 = Mismatch)')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('XNOR Output')
        axes[1, 0].set_ylim([-0.5, 1.5])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency comparison
        sizes = [8, 16, 32, 64, 128, 256]
        power_reductions = [self.calculate_efficiency(s)['power_reduction_percent'] 
                          for s in sizes]
        axes[1, 1].plot(sizes, power_reductions, marker='o', linewidth=2, 
                       markersize=8, color='#667eea')
        axes[1, 1].set_title('Power Reduction vs Input Size')
        axes[1, 1].set_xlabel('Input Size')
        axes[1, 1].set_ylabel('Power Reduction (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bnn_operations_comparison.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'bnn_operations_comparison.png'")
        plt.show()


def main():
    """Demo the BNN operations visualizer"""
    print("=" * 60)
    print("Binary Neural Network Operations Visualizer")
    print("=" * 60)
    
    visualizer = BinaryOpsVisualizer()
    
    # Compare operations
    results = visualizer.compare_operations(size=8)
    
    print("\n--- Operation Results ---")
    print(f"MAC Result: {results['mac_result']:.4f}")
    print(f"XNOR-Popcount: {results['xnor_popcount']}")
    print(f"Approximate MAC from BNN: {results['approximate_mac']:.4f}")
    
    # Calculate efficiency
    efficiency = visualizer.calculate_efficiency(128)
    print("\n--- Efficiency Gains (128-bit operations) ---")
    print(f"Power Reduction: {efficiency['power_reduction_percent']:.2f}%")
    print(f"Area Reduction: {efficiency['area_reduction_percent']:.2f}%")
    print(f"Speedup: {efficiency['speedup']:.2f}x")
    
    # Visualize
    print("\nGenerating visualization...")
    visualizer.visualize_comparison()


if __name__ == "__main__":
    main()
