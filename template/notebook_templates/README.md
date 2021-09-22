
# Notebook Templates

Rather than copy and paste, use the [nbtplt](https://github.com/forforf/pycharm-jupyter-templates) package
to create templates that have the common boiler-plate.

The [nbtplt](https://github.com/forforf/pycharm-jupyter-templates) package is a bare-bones templating engine
purpose built for PyCharm created Jupyter notebooks. It might not work well for other use cases 
(but it shouldn't be too hard to extend it).


The templating itself is pretty user-friendly (IMO), but the generation process is somewhat manual.

## Templating

The README in [nbtplt](https://github.com/forforf/pycharm-jupyter-templates) explains the general idea.

In general there are 4 notebook types
* Source notebooks: Contain the source cells that will be injected into the templates.
* Template notebooks: Contain the boiler-plate and placeholders for the source notebook cells.
* Generated notebooks: The final notebooks after the source cells have been injected into the templates.
* Template Runner notebook: A single notebook for orchestrating the source, templated and generated notebooks.

### Template Runner Notebook

The template runner for this project is located at `template/notebook_templates/model-generator.ipynb`

Executing this notebook will create all the necessary notebooks in the `template/notebook_templates/generated-nb`
directory.

#### Execution Overview

The runner will take the source notebooks and apply an intermediate template 
(`sklearn-tplt` and `keras-tplt` in the diagram below). The output of that 
template then becomes the input for the model template (`model-tplt`). The output
of the model template is the fully hydrated notebook.

```
               sklearn-tplt                model-tplt
sksrc1  ----------> + ---> sklearn-sksrc1 ----> + ---> model-sklearn-sksrc1
sksrc2  ----------> + ---> sklearn-sksrc2 ----> + ---> model-sklearn-sksrc2

                keras-tplt                 model-tplt
kerassrc1 --------> + ---> keras-kerassrc1 ---> + ---> model-keras-kerassrc1
kerassrc2 --------> + ---> keras-kerassrc2 ---> + ---> model-keras-kerassrc2
```

**Pseudo-code**
```python
sk_chain = NbtpltChain('sklearn-', 'sklearn-tplt', ['sksrc1', 'sksrc2'])
sk_chain.generate()
# generate() creates the following files: 
#  sklearn-sksrc1
#  sklearn-sksrc2 

keras_chain = NbtpltChain('keras-', 'keras-tplt', ['kerassrc1', 'kerassrc2'])
keras_chain.generate()
# generate() creates the following files: 
#  keras-kerassrc1
#  keras-kerassrc2

# Second stage uses the generated files
generated_srcs = ['model-sklearn-sksrc1', 'model-sklearn-sksrc2', 'model-keras-kerassrc1', 'model-keras-kerassrc2']
model_chain = NbtpltChain('model-', 'model-tplt', generated_srcs)
model_chain.generate()
# generate() creates the following files:
#  model-sklearn-sksrc1
#  model-sklearn-sksrc2
#  model-keras-kerassrc1
#  model-keras-kerassrc2
                          
```

Final generated notebooks are manually copied/moved from `template/notebook_templates/generated-nb` directory
to the `kss-event-detection-rms/models` directory.


See `model-generator.ipynb` or [nbtplt](https://github.com/forforf/pycharm-jupyter-templates) for more details
on source code implementation.

### Source Notebooks
Source notebooks have two flavors. 
* Source notebooks used as initial source input
* Generated source notebooks that are the output of a source + template

#### Regular Source Notebooks
* found in `template/notebook_templates/template-src`
* have a file naming format of `<short model name>.nbtplt.ipynb`
* short model name examples: `logr`, `svr-rbf`, `tree`, `cnn`, etc

#### Generated Source Notebooks
* found in `template/notebook_templates/generated-nb`
* have a file format of `<group name>-<short model name>.nbtplt.ipynb`
* group name examples `sklearn`, `keras`

See also **Generated Notebooks** section

### Template Notebooks
* found in `template/notebook_templates/template-src`
* have a file naming format of `<short model name>.nbtplt-proxy.ipynb` (Note the `-proxy`)


**Fun Fact**: Template Notebook cell code is indistinguishable from Source Notebook cell code.

What distinguishes a template notebook from a source notebook is whether it is used as the 2nd or 3rd
argument in the `NbtpltChain` constructor. The 2nd argument is the template to be applied, while the
3rd argument is the source data to be injected into the template.

```python
NbtpltChain(<dest path prefix>, <template>, [<source>, <source>, ...])
```

In other words, the template cell and source cell binding is done through the nbtplt id field 
(let's call it `cell-id`). The cell data associated with `cell-id` in the template notebook is replaced
by the cell data associated with `cell-id` in the source notebook. So the difference between a Template
Notebook and a Source Notebook is whether that notebook's cells are being replaced (template) or injected (source).

### Generated Notebooks

* found in `template/notebook_templates/generated-nb`
* Their filenames are assigned a prefix* by `NbtpltChain` class

**Another Fun Fact**: Generated Notebook cell code is also indistinguishable from template and source notebooks.
So Generated Notebooks can also be used as templates our sources.

A generated Notebook is just another notebook, the only difference is it has been generated by the
templating engine.

*The [nbtplt](https://github.com/forforf/pycharm-jupyter-templates) NbtpltChain class is 
(currently) very basic and only allows users to provide a prefix (which can include directory paths)
to the generated notebook filename.
