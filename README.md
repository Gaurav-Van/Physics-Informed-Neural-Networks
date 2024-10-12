# Physics-Informed-Neural-Networks
Exploring the concepts of Physics Informed Neural Networks. Coding a Physics Informed Neural Network to simulate a harmonic oscillator and solve a simple first-order ordinary differential equation.

#### References 
- **Medium Article by Mario Dagrada** - Explains the Core Idea of `PINNs` and `Residual of govering physics equations` in an clear and concise manner. Provides a Hands on experience of PINNs by coding a PINN to solve simple first-order ordinary differential equation.<br>https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4.
- **Physics-Informed Neural Networks (PINNs) - An Introduction - Ben Moseley** - An Inroductory Crash course on PINNs <br> https://www.youtube.com/watch?v=G_hIppUWcsc <br> https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/

## Overview
Physics-Informed Neural Networks (PINNs) are a class of machine learning models that integrate traditional data-driven approaches with the governing physics equations, typically partial differential equations (PDEs). This integration ensures that the model's predictions are both accurate and physically consistent.

`Main Concept of PINNs`: Physics-Informed Neural Networks (PINNs), the loss function combines traditional data loss with the residuals of the governing physics equations (PDEs). By minimizing this combined loss, the neural network ensures its predictions fit the data and adhere to the physical laws. This approach leverages the network’s optimization capabilities to produce accurate and physically consistent results

`Core Idea`: Just Combine traditional loss function with residuals of the physics equation of the problem or environment and optimize the combined loss function. 

```txt
Incorporating the residuals into the traditional error function in a meaningful way
```

```txt
Boundary Loss + Physics Loss
```

## Key Concepts 
- **Data Loss**: The traditional loss function that measures the difference between the predicted values and the actual data.
- **Physics Residuals**: The residuals of the governing physics equations (PDEs) that the model must satisfy.
- **Combined Loss Function**: By minimizing a loss function that combines both data loss and physics residuals, PINNs ensure that their predictions fit the data and adhere to the physical laws.

<hr>

## Explanation 

1. **Neural Networks (NNs) Properties**
    - **Universal Function Approximators**: NNs can approximate any function if they are deep and expressive enough.
    - **Automatic Differentiation (AD)**: Allows easy and efficient computation of derivatives of NN outputs with respect to inputs and model parameters.
  
2. **Residuals in PINNs**
    - **Definition**: The residual is the difference between the left-hand side and the right-hand side of the differential equation when the NN’s output is substituted into it.
    - **Loss Contribution**: Evaluate the residual at collocation points and compute the mean square error (MSE) or any other error method. <ins>_Incorporating the residuals into the traditional error function in a meaningful way. Boundary Loss + Physics Loss_</ins>
      
3. **Boundary Conditions**
    - **Inclusion in Loss Function**: Add boundary conditions to the loss function to ensure a unique solution.
    - **Final Loss Function**: Combine the residual loss and boundary condition loss:Loss=Lossresidual​+Lossboundary​
      
4. **Optimization**
    - **Training**: Adjust the NN’s parameters to minimize the combined loss function.
    - **Outcome**: The NN output satisfies both the differential equation and the boundary conditions, approximating the final solution.
<hr>

### 3 typical Scientific tasks
1. `Forward Simulation`: Forward Simulation is a process where we use known physical laws, initial conditions, and parameters to predict the future state of a system. This is essentially solving the forward problem, where the governing equations (often differential equations) are used to simulate the behavior of the system over time. 
2. `Inversion`: Inversion is the process of determining the underlying parameters or causes of a system from observed data. This is the inverse problem, which is often more challenging than the forward problem because it can be ill-posed (i.e., solutions may not exist, be unique, or be stable).
3. `Equation Discovery`: Equation Discovery involves identifying the underlying mathematical equations that describe a system’s behavior from data. This task leverages machine learning and symbolic regression to find the governing equations that best fit the observed data.

![image](https://github.com/user-attachments/assets/88dbb549-fd3b-48a0-8852-efd3b0f68016)

### Scientific Machine Learning (SciML) 
![image](https://github.com/user-attachments/assets/2f0bfa57-7fff-4ae0-a043-7e880fcad831)
##### Some topics related to SciML
- AI Feynman
- Covariant neural networks
- Hidden physics models
- AlphaFold
- Mining gold from implicit models
- Hamiltonian / Lagrangian neural networks
- **Physics-Informed Neural Networks (PINNs)**
- Physics-Informed Fourier neural operators
- Physics-Informed DeepONets
- **PINNs** on meshes
- Algorithm unrolling
- Universal differential equations
- Learned sub-grid processes
- Solver in the loop
- Learned gradient descent
- Adversarial regularisers
<hr>

## Naive and Physics-Informed Neural Network using Harmonic Oscillations 
![image](https://github.com/user-attachments/assets/b5204b08-e848-4696-b80a-cac4157700f0)
#### Combined Loss Function 
![image](https://github.com/user-attachments/assets/23bc4d53-ad35-4d74-825f-8211a7c25252)

