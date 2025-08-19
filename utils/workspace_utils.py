import random
import numpy as np
from typing import List


def generate_random_positions(num_positions: int, workspace_size: float,
                              min_distance: float, max_attempts: int = 100) -> List[np.ndarray]:
    """Generate random positions with collision avoidance using spatial optimization."""
    positions = []

    for i in range(num_positions):
        best_position = None
        best_min_distance = 0

        for attempt in range(max_attempts):
            # Random position within workspace
            candidate = np.array([
                random.uniform(-workspace_size / 2, workspace_size / 2),
                random.uniform(-workspace_size / 2, workspace_size / 2),
                random.uniform(-0.05, 0.05)
            ])

            if not positions:
                positions.append(candidate)
                break

            # Calculate minimum distance to existing positions
            distances = [np.linalg.norm(candidate[:2] - pos[:2]) for pos in positions]
            min_dist = min(distances)

            if min_dist >= min_distance:
                positions.append(candidate)
                break
            elif min_dist > best_min_distance:
                best_position = candidate
                best_min_distance = min_dist
        else:
            # If no good position found, use the best one
            if best_position is not None:
                positions.append(best_position)
            else:
                # Fallback: place at center with small offset
                fallback = np.array([
                    random.uniform(-workspace_size / 4, workspace_size / 4),
                    random.uniform(-workspace_size / 4, workspace_size / 4),
                    random.uniform(-0.05, 0.05)
                ])
                positions.append(fallback)

    return positions
