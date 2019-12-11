# Machine Learning of Discriminative Gate Locations for Clinical Diagnosis

This repository contains an implementation of the disciminative automated gating algorithms for flow cytometry data described in our paper: 
[Machine Learning of Discriminative Gate Locations for Clinical Diagnosis](https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.23906)
(Disi Ji, Preston Putzel, Yu Qian, Ivan Chang, Aishwarya Mandava, Richard H. Scheuermann, Jack D. Bui, Huan‐You Wang and Padhraic Smyth)

<!--<p align="center">-->
  <!--<img src="figures/gate_plots_with_best_params.png" width="700" title="hover text">-->
<!--</p>-->

<p align="center">
  <img src="figures/testing_notation_plot.png" width="350" title="hover text">
</p>

## Dependencies
To install dependencies in `requirements.txt`:
```
pip3 install -r requirements.txt
```


## How to run the code
We evaluate the proposed approach using both simulated and real data, 
producing classification results on par with those generated via human expertise, 
in terms of both thepositions of the gating boundaries and the diagnostic accuracy.

To run our code:
1. Download the data from the following link:
2. Create a cll directory in the data directory
3. Move the data folders into the cll directory
4. Modify the config file with the same name as the main
5. Run the desired main from the command line

Example

To use non-default experiment configurations, 
modify the correspondig default config file found in the configs folder, 
and change the path_to_yaml variable at the end of each main to match the modified config file.  

For example to run model X with Y settings, 
modify Z config file and run the following from the src directory:

'''
Example code 
'''

## Authors

* **Disi Ji**  - [disiji](https://github.com/disiji)
* **Preston Putzel** -
[pjputzel](https://github.com/pjputzel)


## Publication
If you use this repository in your research, please cite the following paper:

_"Machine Learning of Discriminative Gate Locations for Clinical Diagnosis"_ ([PDF](https://onlinelibrary.wiley.com/doi/epdf/10.1002/cyto.a.23906)).

    @article{ji2019machine,
      title={Machine Learning of Discriminative Gate Locations for Clinical Diagnosis},
      author={Ji, Disi and Putzel, Preston and Qian, Yu and Chang, Ivan and Mandava, Aishwarya and Scheuermann, Richard H. and Bui, Jack D. and Wang, Huan‐You and Smyth, Padhraic},
      journal={Cytometry Part A},
      year={2019}
    }

## License and Contact

This work is released under the MIT License.
Please submit an [issue](https://github.com/disiji/fc_differentiable/issues/new) to report bugs or request changes. 
Contact **Disi Ji** [:envelope:](mailto:disij@uci.edu) for any questions or comments. 


## Acknowledgments
The work in this paper was partially supported byNIH/NCATS U01TR001801 (FlowGate), NSF XSEDE alloca-tion MCB170008, and NIH Commons Credits on CloudComputing CCREQ-2016-03-00006. The content is solely theresponsibility of the authors and does not necessarily repre-sent the official views of the National Institutes of Health.
