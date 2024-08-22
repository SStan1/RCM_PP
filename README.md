# RCM++

The aim of this project is to propose a new algorithm to help the RCM algorithm choose a more appropriate starting point. We call the RCM using this new algorithm RCM++ and compare it in detail with the now commonly used GL_RCM and MIND_RCM. Related research has been published in

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)


## 📁 Project Structure

### 1. `data/`  
This folder stores all the matrix data used in the experiments, divided into two datasets. Each dataset is prepared for different experiments to evaluate various aspects of the project.[Click here to access the data](https://drive.google.com/drive/folders/1sxE3xHsu4hjRvBEK1zm13eohjlZ5ahZD?usp=drive_link)

### 2. `src/`  
This directory contains the implementation of various algorithms, including:
- The novel algorithm proposed in this project
- Traditional algorithms such as **GL** and **MIND**

Additionally, it includes code for three distinct experiments that assess the proposed algorithm in terms of:
- **⏱️ Runtime**
- **📊 Result quality**
- **⚡ Equation-solving speed-up**

### 3. `test/`  
This folder provides a runnable example using two small matrices. It offers an intuitive demonstration of the proposed algorithm in action, making it easy for users to explore its functionality.



# Installation and Setup Guide

Follow the steps below to set up the project and run the example:

1. **Clone the repository to your local machine:**
   ```bash
   git clone <repository_url>
   ```
   Replace `<repository_url>` with the actual repository link.

2. **Create the `data` folder:**
   Navigate to the `BNF` folder and create a new folder called `data`. Place the downloaded input files into this folder.

   Example:
   ```bash
   mkdir BNF/data
   ```

3. **Install the required dependencies:**
   Run the following command to install all the dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the BNF directory to your Python path:**
   To ensure Python can locate your project files, append the `BNF` directory to the system path:
   ```python
   import sys
   sys.path.append('/content/drive/MyDrive/BNF')
   ```

5. **Run the example script:**
   Execute the provided example script using the following command:
   ```bash
   %run /content/drive/MyDrive/BNF/test/Example/Example_solve.py
   ```

6. **View the results:**
   Once the script completes, navigate to the `results` folder to check the output files generated by the script.

