# Deep Q-Learning Algorithm

This project implements the **Deep Q-Learning (DQN)** algorithm using PyTorch and Gymnasium environments (including Box2D environments such as LunarLander).

## 📦 Installation

Install the required dependencies using one of the following methods.

### Option 1 – Using pip

```Bash
pip install swig "gymnasium[box2d]" jupyter numpy torch matplotlib onnx onnxscript onnxruntime
```

### Option 2 – Using uv (pip interface)

```Bash
uv pip install swig "gymnasium[box2d]" jupyter numpy torch matplotlib onnx onnxscript onnxruntime
```

### Option 3 – Using uv (project dependency management)

```Bash
uv add swig "gymnasium[box2d]" jupyter numpy torch matplotlib onnx onnxscript onnxruntime
```

## 🚀 Usage

Launch Jupyter Notebook:

```Bash
jupyter notebook
```
Run the training and start training your agent.


## 📝 Notes

-   Make sure `swig` is installed before installing Box2D dependencies.    
-   Box2D environments may require additional system libraries depending on your OS.
