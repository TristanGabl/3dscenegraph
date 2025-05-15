import numpy as np
from collections import defaultdict

def group_objects_by_transitions(matrix):
    """
    Group objects based on transitions of instance IDs across frames.
    
    Args:
        matrix (np.array): A 2D numpy array where rows represent objects (vertices)
                           and columns represent frames. Each element is an instance ID.
    
    Returns:
        dict: A dictionary where the keys are final object IDs and the values are lists
              of instance IDs that belong to the same object.
    """
    # Step 1: Create a transition map for instance IDs across frames
    transitions = defaultdict(list)

    # Iterate over the matrix to track transitions of instance IDs
    num_frames = matrix.shape[1]
    for vertex in range(matrix.shape[0]):  # Each row represents an object/vertex
        for frame in range(1, num_frames):  # Iterate over frames, start from the second frame
            transitions[matrix[vertex, frame-1]].append(matrix[vertex, frame])

    # Print the transitions to understand the changes in instance IDs across frames
    print("Transitions of object IDs across frames:")
    for obj_id, trans in transitions.items():
        print(f"Object {obj_id} transitions to: {trans}")

    # Step 2: Group the instances into final objects based on transitions
    final_objects = defaultdict(list)
    visited = set()

    # Group based on common transitions
    for obj_id, trans in transitions.items():
        if obj_id not in visited:
            visited.add(obj_id)
            # Check which other object IDs have common transitions
            current_object_group = [obj_id]
            for next_obj in trans:
                if next_obj not in visited:
                    current_object_group.append(next_obj)
                    visited.add(next_obj)
            # Assign a unique object ID for the final group
            final_object_id = len(final_objects) + 1
            final_objects[final_object_id] = current_object_group

    # Print final object groupings
    print("\nFinal Groupings of Objects based on Transitions:")
    for final_id, objects in final_objects.items():
        print(f"Final Object ID {final_id}: {objects}")

    return final_objects

# Example of how to call the function with a matrix
if __name__ == "__main__":
    # Example matrix with 4 objects (rows) and 5 frames (columns)
    matrix = np.array([
        [1, 1, 2, 2, 2],  # Object 1 transitions: 1 -> 2 -> 2 -> 2
        [1, 1, 3, 2, 2],  # Object 1 transitions: 1 -> 2 -> 2 -> 2
        [3, 3, 4, 4, 5],  # Object 2 transitions: 3 -> 4 -> 5
        [6, 6, 6, 6, 7],  # Object 3 transitions: 6 -> 6 -> 7
        [8, 9, 9, 8, 8]   # Object 4 transitions: 8 -> 9 -> 8 -> 8
    ])
    
    grouped_objects = group_objects_by_transitions(matrix)
    print("\nGrouped Objects:", grouped_objects)
