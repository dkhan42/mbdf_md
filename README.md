# mbdf_md
molecular dynamics using MBDF

Script for generating MBDF along with gradients

The representations should either be generated for the entire dataset together using the `generate_mbdf_train` function (you can discard the `norms` array):

```
from MBDF_gradients import generate_mbdf_train

rep, drep, norms = generate_mbdf_train(charges, coordinates, n_jobs = n_jobs)
```

 or separately in the following manner (the `norms` array from the `generate_mbdf_train` function is required) :
 
```
from MBDF_gradients import generate_mbdf_train, generate_mbdf_pred

rep_train, drep_train, norms = generate_mbdf_train(charges_train, coordinates_train, n_jobs = n_jobs)
rep_test, drep_test = generate_mbdf_pred(charges_test, coordinates_test, norms, n_jobs = n_jobs)
```

`charges` and `coordinates` are always arrays containing arrays of molecular charges and coordinates respectively. 

`rep` and `drep` always denote the corresponding representations and their gradients respectively.

`n_jobs` is `-1` by default which means all cores will be used.
