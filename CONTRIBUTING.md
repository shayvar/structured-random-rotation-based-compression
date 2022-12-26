# Contributing to Structured Random Rotation-based Compression


We welcome contributions from the community and first want to thank you for taking the time to contribute!

Please familiarize yourself with the [Code of Conduct](https://github.com/vmware/.github/blob/main/CODE_OF_CONDUCT.md) before contributing.


* Before you start working with Structured Random Rotation-based Compression, please read our [Developer Certificate of Origin](https://cla.vmware.com/dco). All contributions to this repository must be signed as described on that page. Your signature certifies that you wrote the patch or have the right to pass it on as an open-source patch.

## Ways to contribute

We welcome many different types of contributions and not all of them need a Pull request. Contributions may include:

* New features and proposals
* Documentation
* Bug fixes
* Issue Triage
* Answering questions and giving feedback
* Helping to onboard new contributors
* Other related activities

## Getting started

### Installation

`$ pip install srrcomp`

If the message "Using torch implementation for Hadamard and bit packing" appears when importing srrcomp on a GPU machine, try installing `srrcomp` from source by:

`$ python setup.py install`

### Testing

Execute from \tests folder:
`$ python dme_test.py`

Use `$ python dme_test.py -h` to get the test options 

## Contribution Flow

This is a rough outline of what a contributor's workflow looks like:

* Make a fork of the repository within your GitHub account
* Create a topic branch in your fork from where you want to base your work
* Make commits of logical units
* Make sure your commit messages are with the proper format, quality and descriptiveness (see below)
* Push your changes to the topic branch in your fork
* Create a pull request containing that commit

We follow the GitHub workflow and you can find more details on the [GitHub flow documentation](https://docs.github.com/en/get-started/quickstart/github-flow).

Before submitting your pull request, we advise you to use the following:


### Pull Request Checklist

1. Check if your code changes will pass both code linting checks and unit tests.
2. Ensure your commit messages are descriptive. We follow the conventions on [How to Write a Git Commit Message](http://chris.beams.io/posts/git-commit/). Be sure to include any related GitHub issue references in the commit message. See [GFM syntax](https://guides.github.com/features/mastering-markdown/#GitHub-flavored-markdown) for referencing issues and commits.
3. Check the commits and commits messages and ensure they are free from typos.

## Reporting Bugs and Creating Issues

For specifics on what to include in your report, please follow the guidelines in the issue and pull request templates when available.


## Ask for Help

The best way to reach us with a question when contributing is to ask on:

* The original GitHub issue
* The developer mailing list


## Additional Resources

Academic publications:

- Shay Vargaftik, Ran Ben-Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben-Itzhak, and Michael Mitzenmacher. ["DRIVE: One-bit Distributed Mean Estimation."](https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html) Advances in Neural Information Processing Systems 34 (2021): 362-377.

- Shay Vargaftik, Ran Ben Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben Itzhak, and Michael Mitzenmacher. ["EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning."](https://proceedings.mlr.press/v162/vargaftik22a.html) In International Conference on Machine Learning, pp. 21984-22014. PMLR, 2022.

Also, see the following blog for a high-level overview: 
["Pushing the Limits of Network Efficiency for Federated Machine Learning"](https://octo.vmware.com/pushing-the-limits-of-network-efficiency-for-federated-learning/)


