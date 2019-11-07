# Machine Learning of Discriminative Gate Locations for Clinical Diagnosis

This repository contains an implementation of the disciminative automated gating algorithms for flow cytometry data described in our paper: https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.23906



## Dependencies

* matplotlib 3.0.3 or higher 
* numpy 1.16.4 pr or higher
* pandas 0.25.1 or higher
* pytorch 3.7 or higher
* pyyaml 5.1.2
* sklearn 0.21.1 or higher
* scipy 1.2.1 or higher
* seaborn 0.9.0 or higher (only needed for synthetic data plotting code)


## How to run the code
We provide default configuration files which should run out of the box. To do so navigate to the src directory and run the desired main. To use non-default experiment configurations modify the correspondig default config file found in the configs folder, and change the path_to_yaml variable at the end of each main to match the modified config file. For example to run model X with Y settings, modify Z config file and run the following from the src directory:

'''
Example code 
'''

## Authors

* **Disi Ji**  - [disiji](https://github.com/disiji)
* **Preston Putzel** -
[pjputzel](https://github.com/pjputzel)



## Acknowledgments

* Collaborators go here
* Funders go here (see paper acknowledgements section)


