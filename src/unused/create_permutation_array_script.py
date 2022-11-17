from pathlib import Path
import numpy as np

def create_permutation_array_within(
    project_root: str,
    design: str,
    permutations: int = 1000
):

    n_runs = 8

    n_conditions = 8 if design == "expanded" else 2
    values = np.array(range(n_runs * n_conditions))
    groups = [[x] * n_conditions for x in range(n_runs)]
    groups = [item for items in groups for item in items]
    permutation_array = np.zeros([permutations,len(groups)])

    for perm in range(permutations):
        for index in np.unique(groups):
            mask = groups==index
            values[mask] = np.random.permutation(values[mask])

        permutation_array[perm,:] = values.T

    permutation_array = permutation_array
    
    np.savetxt(f"{project_root}/data/utils/{design}_permutation_within.txt",permutation_array, fmt='%i')


project_root=f"{Path.home()}/implied_motion"

create_permutation_array_within(
    project_root=project_root,
    design='compact',
    permutations=1000
)

create_permutation_array_within(
    project_root=project_root,
    design='expanded',
    permutations=1000
)