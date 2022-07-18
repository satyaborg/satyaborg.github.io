---
layout: post
title: Accessing Virtual environments in Jupyter Notebook
tags: [jupyter, python]
---

Virtual environments are great for compartmentalizing dependencies for each project. Jupyter notebooks are great for rapid prototyping. How to combine the two?

One of the very first things when starting a new Python project should be creating a new virtual environment. They help maintain consistency, keep dependencies isolated and mitigate version conflicts among other things.

Let's start by creating a new virtual environment called `venv` for Python and activate the same as follows,

```bash
virtualenv -p python3 venv
source venv/bin/activate
```

Note: We added an additional flag `-p` to ensure that Python 3  is the default interpreter in our environment since Python 2 has reached its [end of life](https://pythonclock.org/) (goodbye ol' friend).

Once we are inside the `virtualenv` we can now install all our dependencies via `pip install <package>` or `pip install -r requirements.txt` if we happen to have a `requirements.txt` file for our project.

Now for the interesting bit: To make this `venv` available as a kernel in our Jupyter notebooks, we simply need to install `ipykernel` within the virtual environment and then *register* the same as follows,

```bash
pip install ipykernel
ipython kernel install --user --name=<foo-kernel>
```

Now whenever we launch a new jupyter notebook instance, we would be able to select `foo-kernel` beside the main global python interpreter (usually located in `/usr/bin/python3`) by navigating to "Kernel" and then clicking on "Change kernel" in our Jupyter notebook.

An alternative to the same, abeit less graceful, is to install Jupyter notebook in *each* environment seperately and launch it from within the environment for the packages to be made available.
