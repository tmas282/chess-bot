# Chess AI Bot

A clean, from-scratch Python implementation of a chess-playing bot built using PyTorch, core Machine Learning and Conventional Advanced Algorithmic principles. 

This project was developed as a deep-dive learning exercise to thoroughly understand neural network architectures, policy/value estimation, and state representation in complex, adversarial board games.

## 🚀 Quick Setup

### Method 1: Local Setup (Using uv)

This project uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver, to manage dependencies and virtual environments.

#### Prerequisites:

Ensure you have Python 3.10+ and `uv` installed. If you don't have `uv` yet, install it via:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Clone the repository:

```bash
git clone https://github.com/tmas282/chess-bot.git
cd chess-bot
```

#### Create a virtual environment:
Using uv, you can spin up an optimized virtual environment instantly:

```bash
uv venv
```

#### Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

#### Install the project packages:

```bash
uv sync
```

---

### Method 2: Containerized Setup (Using Docker)
If you prefer to keep your host machine clean or want a drop-in environment where everything "just works," you can use the provided multi-stage Dockerfile.

#### Build the Docker Image:
From the root of the repository, build the optimized slim image:

```bash
docker build -t chess-bot .
```
#### Access the Interactive Docker Shell
To hop inside the container and manually interact with the files, run tests, or execute arbitrary Python scripts:

```bash
docker run --gpus all -it chess-bot /bin/bash
```
Once inside the shell, the virtual environment is already loaded into your PATH, so you can run Python scripts directly (e.g., python your_script.py).

<br/>

---

<br/>

> **💡 Important Project Note**
>
> **The core architecture, engine integration, training logic, and game-playing algorithms of this repository were designed and written entirely by hand without AI assistance.**
> 
> AI tools were utilized **exclusively** for structuring this documentation (`README.md`) and configuring the `uv` virtual environment setup. This strict separation was maintained to guarantee a genuine, ground-up learning experience of PyTorch and foundational machine learning principles.
