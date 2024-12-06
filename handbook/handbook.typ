#import "lib/template.typ": main
#import "lib/simpleTable.typ": simpleTable
#import "lib/codeBlock.typ": codeBlock
#show: doc => main(
  title: [
    Machine Learning
  ],
  version: "v0.1.",
  authors: (
    (name: "Melany Argandoña", email: "melany.argandona@fundacion-jala.org"),
  ),
  abstract: [
    This is a collection of notes and thoughts that I've been taking while learning about machine learning.
    It is based on the *"Machine Learning"* specialization from Coursera by _Andrew Ng_ as well as the lessons and labs from our course at *Fundación Jala*.
  ],
  doc,
)
= Day 1: First Assessment

1. Given the following matrices:

#figure(
  image("./images/matrix.jpeg", width: 100%)
)

#pagebreak()

2. Write the functions that correspond to the following charts.

#figure(
  image("./images/ecuation.jpeg", width: 80%)
)

#pagebreak()

= Week 1: Introduction to Machine Learning

== Section 2: Supervised vs. Unsupervised Machine Learning

=== Video 1: What is machine learning?
#emph([
  "A field of study that gives computers the ability to learn without being explicitly programmed."
])

#h(1fr) — Arthur Samuel.

The more opportunities you give a learning algorithm, the better its performance will be.

#text(weight: "bold")[Types of learning]

The two main types of learning are:

#text(weight: "bold")[Supervised learning:] This is the most widely used type of learning in applications, experiencing rapid advancements and innovation.

#text(weight: "bold")[Unsupervised learning:] This method is used for finding hidden patterns in data without labeled examples.

#text(weight: "bold")[Learning Algorithms]

The most commonly used types of learning algorithms today are:
Supervised learning, Unsupervised learning and Recommender systems.

=== Video 2: Supervised learning part 1
#text(weight: "bold")[Supervised Learning]
Algorithms that learn mappings from X to Y or from input to output.

#figure(
  image("./images/c1_w1_s2_v02_input_output.png" )
)

You provide the algorithm with examples from which it learns.  
It includes correct answers: the correct label.

#text(weight: "bold")[Examples Supervised Learning]

First, the model is trained with examples of inputs X and the correct answers (labels).  
Once the model has learned pairs of input (x) and output (y), it can take a new input X that it hasn’t seen before and attempt to provide the appropriate corresponding output.

#figure(
  image("./images/c1_w1_s2_v02_examples_input_output.png" ),
  caption: [
    Examples of type supervised learning.
  ]
)

Example:

The x-axis represents the size of the house in square feet, and the y-axis represents the price based on the size of the house in thousands of dollars. A friend wants to know how much his 750-square-foot house is worth.  
The algorithm will first check if a straight line can be drawn through most of the data points, which suggests the house could be worth $dollar$ 150,000. However, to get a more precise answer, it might use a curve instead of a line, estimating the house to be worth $dollar$  200,000. 

#figure(
  image("./images/c1_w1_s2_v02_example_price_house.png" ),
  caption: [
    Examples of type supervised learning using a stimation price of house.
  ]
)

The algorithm systematically chooses the line, curve, or other fitting shape that best matches the data.

The previous example uses a specific type of supervised learning called #text(weight: "bold")[Regression] .

 #text(weight: "bold")[Regression] : Predicting a number from an infinite number of possible numbers, like housing prices in the example, which could be 150,000, 70,000, 183,000, or any other intermediate number.

=== Video 3: Supervised learning part 2

 #text(weight: "bold")[Classification] : Predicts categories, which may not be numerical. It predicts a small, limited set of possible output categories.

 Maps input x to output Y, where the algorithm learns from the "correct answers."
 
 If the categories are numerical, there will never be an intermediate value. For example, if we have 1 and 2, a result of 1.5 will never appear.
 
 The input values may consist of one or more variables to predict the output.

Example:

The example aims to classify whether a tumor is benign or malignant. The x-axis represents the first category, which would be tumor size, and the y-axis represents the patients' age, with only two values representing malignant (x) or benign (o). 

#figure(
  image("./images/c1_w1_s2_v03_example_cancer.png" ),
  caption: [
    Examples of type supervised learning using a detect cancer.
  ]
)


Classification differs from regression because it attempts to predict from a small number of possible outcomes or categories, while regression tries to predict any number, an infinite set of possibilities.

The learning algorithm can find a boundary that separates malignant from benign tumors. The algorithm must decide how to draw a boundary line across the data points. This helps doctors determine whether a tumor is benign or malignant.

The two main types of supervised learning are  #text(weight: "bold")[Regression] and  #text(weight: "bold")[Classification] .

=== Video 4: Unsupervised learning part 1

#figure(
  image("./images/c1_w1_s2_v04_cancer_Example.png" ),
  caption: [
    Examples of type supervised learning using a detect cancer.
  ]
)

In unsupervised learning, each example was associated with an output label, like benign or malignant, shown by circles and crosses.

If we receive data not associated with any output label, let’s say we’re given data on patients, their tumor sizes, and ages. But we don’t know if the tumor was benign or malignant, so the dataset looks like the example on the right.  
We aren’t asked to diagnose if a tumor is benign or malignant because no labels are provided. 

In this dataset, our task is to find some structure or pattern or simply something interesting in the data.

Therefore, we might decide there are two distinct groups. This type of unsupervised learning is called a 
#text(weight: "bold")[clustering algorithm] .

#text(weight: "bold")[Unsupervised Learning] : The algorithm is not supervised.

Instead of giving a correct answer for each input, we ask the algorithm to discover on its own what might be interesting or what patterns or structures may be in this dataset.

The algorithm might determine that the data can be assigned to two different groups or clusters.

Example:

Many companies have enormous customer databases and can use this data to automatically group customers into different market segments to serve them more effectively.

#figure(
  image("./images/c1_w1_s2_v04_example_IA.png" ),
  caption: [
    Examples of type supervised learning using a detect cancer.
  ]
)

Market segmentation found a few distinct groups of people. The primary motivation of one group is to gain knowledge to develop their skills. Another group is primarily motivated by career advancement. And another group wants to keep up with the impact of AI on their field of work.

#text(weight: "bold")[Clustering algorithm:] A type of unsupervised learning algorithm that takes unlabeled data and automatically tries to group it into clusters.

=== Video 5: Unsupervised learning part 2

#text(weight: "bold")[Reminder from the previous video:]

#text(weight: "bold")[Supervised Learning:] Data includes inputs x with corresponding output labels y.  

#text(weight: "bold")[Unsupervised Learning:] Data includes only inputs x, without output labels y. The algorithm must find some structure, pattern, or interesting feature in the data.

#text(weight: "bold")[Types of unsupervised learning algorithms:] 

- *Clustering algorithm:* Groups similar data points.
- *Anomaly detection:* Used to detect unusual events.
- *Dimensionality reduction:* Takes a large dataset and compresses it into a much smaller dataset while preserving as much information as possible.

== Section 4: Regression Model

=== Video 1: Linear regression model part 1

*Linear regression model: *Fit a straight line to the data.

*Problem:*
Predict the price of a house based on the size of the house. A dataset is used that includes the size and prices of houses in Portland, a city in the United States. The horizontal axis represents the house size in square feet, and the vertical axis represents the price of a house in thousands of dollars.

#figure(
  image("./images/c1_w1_s4_v01_example:house_price.png" )
)

The data points of several houses from the dataset are plotted. Each of these small crosses represents a house with the size and price it was recently sold for.

The house is 1250 square feet. How much do you think this house could be sold for? One thing you could do is create a linear regression model from this dataset. Your model will fit a straight line to the data, which might look like this.

The straight line fits the data, the house has 1250 square feet, and it intersects the line that best fits at around $dollar$ 220,000. This is an example of what is called a supervised learning model.

#figure(
  image("./images/c1_w1_s4_v01_example:house_linea.png" )
)

This is supervised learning because the model is first trained by providing data with the correct answers.

This linear regression model is a specific type of supervised learning model. It is called a regression model because it predicts numbers as outputs, just like the prices in dollars.

Any supervised learning model that predicts a number such as 220,000 or 1.5 or less than 33.2 addresses what is known as a regression problem.

The standard notation to indicate the output variable being predicted.

#figure(
  image("./images/c1_w1_s4_v01_example:house_i.png" )
)

The superscript i in parentheses is not an exponentiation. It is simply an index of the training set, referring to row i of the table.


=== Video 2: Linear regression model part 2

Supervised learning: includes input features (such as the size of a house) and output targets (such as the price of the house). The goal is for the algorithm to learn a function \( f \), known as a model, that can make predictions \( \hat{y} \) (estimates) based on new inputs \( x \).


#figure(
  image("./images/c1_w1_s4_v02_terms.png" )
)

The *linear regression* model is introduced, which is a simple linear function represented as \( f(x) = wx + b \), where \( w \) and \( b \) are parameters that the algorithm adjusts to minimize the difference between the predictions \( \hat{y} \) and the actual values \( y \) in the training data. This adjustment uses a *cost function*, an essential concept for evaluating how well the model fits the data.


#figure(
  image("./images/c1_w1_s4_v02_function_Example.png" )
)

Although we start with a linear function for simplicity, it is possible to work with more complex models, such as nonlinear functions.


=== Video 3: Cost Function Formula

* 1. Linear Regression Model*

It is a linear function $\( f_{w,b}(x) = w \cdot x + b \)$, where:
- \( w \): slope (determines the incline of the line).
- \( b \): y-intercept (vertical shift).

The values of \( w \) and \( b \) are adjusted so that the line fits the training data as closely as possible.

#figure(
  image("./images/c1_w1_s4_v03_function_explain.png" )
)

*2. Errors and Predictions*
Each training data point has a feature (\( x^i \)) and a target (\( y^i \)).
The model's prediction is \( \hat{y}^i = $f_{w,b}(x^i) \)$.
The *error* is the difference between the actual value (\( y^i \)) and the predicted value (\( \hat{y}^i \)).

#figure(
  image("./images/c1_w1_s4_v03_function_found_line.png" )
)

=== 3. Cost Function

It measures how well the model fits the training data.
It uses the *mean squared error*:
\[
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \big(f_{w,b}(x^i) - y^i\big)^2
\]
- \( m \): number of training examples.
- The division by \( 2m \) simplifies later mathematical calculations.

#figure(
  image("./images/c1_w1_s4_v03_function_cost.png" )
)

*4. Purpose of the Cost Function*

A large value of \( J(w, b) \) indicates that the model does not fit the data well.
The goal is to find the values of \( w \) and \( b \) that minimize \( J(w, b) \).

The cost function, called *mean squared error*, is commonly used in regression problems due to its effectiveness in various applications.

=== video 4: Cost Function Intuition

Use of the *cost function* in the context of linear regression to determine the best parameters that fit a model to a training dataset.

*Definition of the Cost Function*

#figure(
  image("./images/c1_w1_s4_v04_function_cost_definition.png" )
)

- The *cost function* \( J(w, b) \) measures the average squared error between the model's predictions and the actual values from the data.
- Mathematically, it is expressed as:

  \[
  J(w, b) = \frac{1}{2m} \sum_{i=1}^m \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
  \]
  where:
  - \( m \): number of examples in the dataset.
  - \( $f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b$ \): model prediction.
  - \( y^{(i)} \): actual value associated with \( x^{(i)} \).

*Simplified Example (without \( b \))*

#figure(
  image("./images/c1_w1_s4_v04_function_cost_example.png" )
) 

- To simplify, we set \( b = 0 \), reducing the model to:
  \[
  $f_{w}(x) = w \cdot x$
  \]
- The cost function now depends only on \( w \): \( J(w) \).

*Visualization of Model Fit*

#figure(
  image("./images/c1_w1_s4_v04_function_cost_group.png" )
) 

1. *Training Set*:
   Three points: \( (1, 1) \), \( (2, 2) \), \( (3, 3) \).
   
2. *Graph of \( f_w(x) \)*:
   - When \( w = 1 \):
     - \( f_w(x) = x \), a straight line that passes exactly through all points.
     - \( J(1) = 0 \), since there is no squared error.
   - When \( w = 0.5 \):
     - The line has a smaller slope, generating squared errors.
     - \( J(0.5) \approx 0.58 \).
   - When \( w = 0 \):
     - The line is horizontal and does not fit the data well.
     - \( J(0) \approx 2.33 \).
     
3. *Graph of \( J(w) \)*:
   - Shows how the cost changes as \( w \) varies.
   - The minimum of \( J(w) \) occurs at \( w = 1 \), the optimal value.

*General Interpretation*

- Each value of \( w \) generates a line \( f_w(x) \) with an associated cost \( J(w) \).
- The goal is to minimize \( J(w) \) to find the optimal \( w \).
- In this example, the \( w \) that minimizes \( J(w) \) is \( w = 1 \), providing the best fit to the data.

Linear regression uses the cost function to quantify prediction error and search for parameters (\( w \) and \( b \)) that minimize this error. This process ensures that the model fits the training data optimally, providing accurate predictions.

=== Video 5: Visualizing the Cost Function

Visualization of the *cost function J(w, b)* in the context of a linear regression model with two parameters, \( w \) and \( b \). The goal is to minimize this cost function to find the optimal values of \( w \) and \( b \) that best fit the model to the data.

#figure(
  image("./images/c1_w1_s4_v05_function_cost_model.png" )
)  

1. *Model and Cost Function*:
   - The model \( f(x) = wx + b \) predicts values based on \( x \), \( w \), and \( b \).
   - The cost function \( J(w, b) \) measures the error between the model's predictions and the actual data.

2. *3D Visualization*:

#figure(
  image("./images/c1_w1_s4_v05_image_3d.png" )
)  

   - When only \( w \) is considered, \( J(w) \) takes the form of a U-shaped curve (similar to a "bowl").
   - By including \( b \), \( J(w, b) \) becomes a 3D surface in the shape of a "bowl" or "hammock." Each point on this surface represents a value of \( J \) for a particular pair of \( w \) and \( b \).

#figure(
  image("./images/c1_w1_s4_v05_image_grafics.png" )
) 

3. *Contour Plots*:
   - Alternatively, the 3D surface can be represented in 2D as a *contour plot*.
   - Each ellipse or contour shows points with the same value of \( J \).
   - The central point of the smallest contour corresponds to the minimum of \( J(w, b) \), i.e., the optimal values of \( w \) and \( b \).

4. *Interpretation of the Plots*:
   - Contour plots help visualize how \( w \) and \( b \) affect \( J \) in two dimensions.
   - This makes it easier to identify the minimum of the cost function.

=== video 6: Visualization Examples

How the choices of parameters \( w \) and \( b \) in a linear regression model affect the cost function \( J(w, b) \) and how these choices reflect on the fitted line \( f(x) \). Visual examples are used to illustrate different combinations of \( w \) and \( b \), showing how prediction errors impact the cost value:

1. *Visualization of the relationship between \( w \) and \( b \):*

#figure(
  image("./images/c1_w1_s4_v06_image_example_1.png" )
) 

   - Different combinations of \( w \) and \( b \) generate straight lines \( f(x) \) that may fit the dataset better or worse.
   - If the line doesn't fit well, the cost \( J(w, b) \) will be high, and the point on the contour plot will be far from the center of the concentric ellipses, where the cost is minimum.

2. *Practical Examples:*

#figure(
  image("./images/c1_w1_s4_v06_image_example_2.png" )
) 

   - Lines with different values of \( w \) and \( b \) are visually analyzed. Lines that don't fit well have higher cost values.
   - The minimum cost point on the contour plot corresponds to the optimal combination of \( w \) and \( b \), which defines the best-fit line.


== Section 6: Train the model with gradient descent

=== Video 1: Gradient Descent

The *gradient descent algorithm* is a key technique for minimizing cost functions in machine learning, including linear regression and more complex models like deep neural networks.

*Key Points:*

#figure(
  image("./images/c1_w1_s6_v01_function_lineal.png" )
) 

1. *Objective*: Minimize a cost function \( J(w, b) \) by selecting optimal values for the parameters \( w \) and \( b \).  
2. *Method*:  
   - Start with initial values for \( w \) and \( b \) (commonly set to 0).  
   - In each iteration, adjust \( w \) and \( b \) by moving in the steepest downhill direction on the cost function graph.  
   - Repeat until reaching a local minimum (a valley on the \( J \) graph).  
3. *Property*:  
   - Depending on the starting point, gradient descent may converge to different local minima.  

4. *Visual Example*:  
   - The graph of \( J(w, b) \) is compared to a mountainous terrain. Gradient descent simulates controlled steps towards the nearest valley bottom.  
5. *Importance*:  
   - Gradient descent is essential not only in linear regression but also in training more complex deep learning models.


=== Video 2: Implementing Gradient Descent

#figure(
  image("./images/c1_w1_s6_v02_function.png" )
) 

Gradient descent is an iterative algorithm that adjusts parameters \( w \) and \( b \) to minimize a cost function \( J(w, b) \). At each step, the parameters are updated using the formula:  
\[
w := w - \alpha \frac{\partial J}{\partial w}, \quad b := b - \alpha \frac{\partial J}{\partial b}
\]
where \( \alpha \) is the learning rate, a small value controlling the step size in the downhill direction.

*Key Points:*
1. *Simultaneous Updates*:  
   - The values of \( w \) and \( b \) are calculated and updated simultaneously to ensure consistency. This involves computing temporary values (\( \text{temp}_w, \text{temp}_b \)) before assigning them to \( w \) and \( b \).  
2. *Incorrect Implementation*:  
   - Updating \( w \) before calculating \( b \) leads to inconsistent results, as the new \( w \) affects \( b \)'s calculation.  
3. *Derivatives*:  
   - Derivatives indicate the steepest downhill direction of \( J \), and along with \( \alpha \), they determine the magnitude of parameter adjustments.  
4. *Convergence*:  
   - The algorithm repeats until \( w \) and \( b \) change minimally between iterations, indicating a local minimum.


=== Video 3: Gradient Descent Intuition

Gradient descent is an algorithm used to minimize cost functions by adjusting model parameters (like \( w \) and \( b \)). The *learning rate (\( \alpha \))* determines the step size for updates, while *partial derivatives* guide adjustments to reduce the cost.

*Key Insights:*

#figure(
  image("./images/c1_w1_s6_v03_example.png" )
) 

1. *Positive Derivative*:  
   - If the slope of the cost function (\( J(w) \)) at a point is positive, \( w \) decreases by subtracting \( \alpha \cdot \frac{d}{dw}J(w) \), moving towards the minimum.  
2. *Negative Derivative*:  
   - If the slope is negative, \( w \) increases, as subtracting a negative number is equivalent to adding, also moving \( w \) towards the minimum.  
3. *Slopes and Minima*:  
   - Gradient descent adjusts values to reduce \( J(w) \), progressing towards the nearest minimum.  

A very small \( \alpha \) slows convergence, while a large \( \alpha \) may cause oscillation or divergence, preventing convergence.

=== Video 4: Learning Rate

*Summary of Learning Rate and Gradient Descent*

#figure(
  image("./images/c1_w1_s6_v04_resume.png" )
) 

This section explains the importance of the learning rate (\(\alpha\)) in the gradient descent algorithm, a method for minimizing a cost function \( J(w) \). Key points include:

1. *Impact of Learning Rate*:  
   - If \(\alpha\) is *too small*, update steps will be tiny, making the algorithm extremely slow to reach the minimum.  
   - If \(\alpha\) is *too large*, the algorithm may overshoot the minimum and oscillate or diverge, preventing convergence.  

Image: c1_w1_s6_v04_impact_point.png  

2. *Behavior at a Local Minimum*:  
   - If \( w \) is already at a local minimum, the slope (\(\nabla J\)) is zero, so \( w \) remains unchanged. This ensures that gradient descent stays at the solution.  

Image: c1_w1_s6_v04_education_gradiant.png  

3. *Automatic Step Adjustment*:  
   - As the algorithm approaches the minimum, slopes (\(\nabla J\)) decrease, naturally reducing update step sizes, even with a fixed learning rate.  

Image: c1_w1_s6_v04_education_automatic.png  

4. *General Application of Gradient Descent*:  
   - This algorithm minimizes any cost function, not just the mean squared error used in linear regression. In future steps, it will combine with specific functions to train models like linear regression.

Selecting the right \(\alpha\) is critical for the efficiency and success of gradient descent.

=== Video 5: Gradient Descent for Linear Regression

This section explains how to train a linear regression model using a cost function based on squared error and the gradient descent algorithm.

#figure(
  image("./images/c1_w1_s6_v05_past_video.png" )
) 

The derived formulas for calculating the gradients of parameters \( w \) (slope) and \( b \) (intercept) are explained, showing they result from differential calculus.

#figure(
  image("./images/c1_w1_s6_v05_formulas.png" )
) 

The model fits a straight line to training data by iteratively minimizing the cost function.

#figure(
  image("./images/c1_w1_s6_v05_algorim.png" )
) 

The cost function for linear regression is convex (bowl-shaped), ensuring gradient descent always converges to the global minimum if the learning rate is appropriate.

#figure(
  image("./images/c1_w1_s6_v05_example_1.png" )
) 

#figure(
  image("./images/c1_w1_s6_v05_example_2.png" )
) 

=== Video 6: Running Gradient Descent

This section demonstrates how gradient descent trains a linear regression model.  

#figure(
  image("./images/c1_w1_s6_v06_example_1.png" )
) 

Visual examples show how parameters \( w \) and \( b \) gradually change with each iteration, reducing the cost function and better fitting the model line to data. It starts with initial parameters (\( w = -0.1 \), \( b = 900 \)) and shows improvement until the global minimum is reached.

#figure(
  image("./images/c1_w1_s6_v06_batch.png" )
) 

= Week 2: Regression with multiple input variables

== Section 1: multiple linear regression

=== Video 1: Multiple Features

*Multiple Linear Regression with Multiple Features*

*Introduction*
- In the original linear regression model, only one feature \( x \) was used to predict $\( y \): \( $f{w,b}(x) = wx + b $\)$.

#figure(
  image("./images/w2_s1_v1_one_characteristic.png" )
) 

- Now, it extends to work with multiple features (\( X_1, X_2, ..., X_n \)).

#figure(
  image("./images/w2_s1_v1_multiple_characteristic.png" )
) 

*New Notations*

1. **Features**:
   - \( X_j \): Represents the \( j \)-th feature (\( j \) from 1 to \( n \)).
   - \( X^{(i)} \): A vector containing all features for the \( i \)-th training example.
   - $\( X^{(i)}_j \)$: Value of the \( j \)-th feature in the \( i \)-th example.
2. **Model Parameters**:
   - \( W \): A vector containing the weights (\( W_1, W_2, ..., W_n \)).
   - \( b \): The intercept, a single number.
3. **Model with Multiple Features**:
   $\[
   f_{w,b}(X) = W_1X_1 + W_2X_2 + \l $dots + W_nX_n + b$
   \]$

#figure(
  image("./images/w2_s1_v1_new_notation.png" )
) 

*Concrete Example*

- To predict a house price (\( y \)):
  $ \[
   f_{w,b}(X) = 0.1X_1 + 4X_2 + 10X_3 - 2X_4 + 80
   \]$
   - \( X_1 \): House size (\(+100 \, \text{USD per square foot}\)).
   - \( X_2 \): Number of bedrooms (\(+4000 \, \text{USD per bedroom}\)).
   - \( X_3 \): Number of floors (\(+10,000 \, \text{USD per floor}\)).
   - \( X_4 \): House age (\(-2000 \, \text{USD per year}\)).

#figure(
  image("./images/w2_s1_v1_example.png" )
) 

*Compact Form Using Vectors*
1. \( W \) and \( X \) are vectors:
   - \( W = [W_1, W_2, \ldots, W_n] \).
   - \( X = [X_1, X_2, \ldots, X_n] \).
2. The model can be rewritten using the **dot product**:
  $ \[
   f_{w,b}(X) = W \cdot X + b
   \]$
   - Dot product: \( W \cdot X = W_1X_1 + W_2X_2 + \ldots + W_nX_n \).

*Key Terms*
- **Multiple Linear Regression**:
  - Uses multiple features to predict the target value.
  - Different from **univariate regression** (1 feature) and "multivariate regression" (which refers to something else).
- **Vectorization**:
  - A technique to efficiently implement models with multiple features.

=== Video 2: Vectorization Part 1

Vectorization is a powerful technique that simplifies and accelerates algorithm implementation, particularly in machine learning contexts.

*Key Concepts*
1. **Definition of Vectorization**:
   - The use of vectorized mathematical operations to replace explicit loops in code.
   - Leverages optimized libraries and advanced hardware, such as multi-core CPUs or GPUs.

2. **Example of Vectorization**:
   - Given a parameter vector \( w \) and a feature vector \( x \), calculating a function \( f \) without vectorization requires explicit loops or manual multiplication and addition.
   - With **NumPy**, a Python linear algebra library, the `np.dot(w, x)` method performs the dot product efficiently, combining multiplication and summation.

3. **Benefits of Vectorization**:
   - **Shorter Code:** Reduces multiple lines to a single line, improving readability and maintainability.
   - **Higher Speed:** Vectorized implementations utilize parallel hardware, resulting in significantly shorter execution times.

4. **Non-Vectorized Implementations**:
   - **Basic Code:** Manually writing each operation.
   - **Using Loops:** Slightly better but inefficient for large \( n \) (e.g., \( n = 100,000 \)).

5. **Vectorized Implementation**:
   - Use optimized functions like `np.dot` and leverage modern hardware.
   - Example:
     ```python
     import numpy as np

     w = np.array([1, 2, 3])
     x = np.array([4, 5, 6])
     b = 1
     f = np.dot(w, x) + b
     print(f)
     ```

*Comparative Advantages*
- **Performance:** NumPy uses parallel instructions on modern hardware, speeding up calculations.
- **Simplicity:** Simplifies development and debugging.
- **Scalability:** Ideal for large-scale computations.

In summary, vectorization not only reduces development time by minimizing code but also optimizes performance, fully leveraging modern hardware capabilities.






















== Section 2: Supervised vs. Unsupervised Machine Learning

To recap, simple linear regression with one variable is given by:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Model*], [
    $ f_(w,b)(x) = w x + b $
  ],
  [*Parameters*], [
    $ w, b $
  ],
  [*Cost Function*], [
    $ J(w, b) = 1/(2m) sum_(i=1)^m ( f_(w,b)( x^(\(i\)) ) - y^(\(i\) ) )^2 $
  ],
  [*Objective*], [
    $ min_(w,b) J(w, b) $
  ],
)

This is the simplest form of linear regression, and we are given a dataset:

$ (X, Y) $

Where $X$ is a vector of features and $Y$ is a vector of labels.

#figure(
  image("./images/2024-10-30-simple-regression.png"),
  caption: [
    I made this diagram using _excalidraw_, that, in my head, represents what we are trying to do. $x^(\(i\))$ is the $i$-th training example, and $y^(\(i\))$ is the $i$-th training label.
  ]
)

=== Cost Function

One interesting observation is how the cost function $J(w)$ changes as we change the value of $w$. For example, the code below plots the cost for a simple target:

$ f_(w, b = 0) = w x $

Notice that for convenience, we are using $b = 0$, so our target is simply $f_(w) = w x$.

#codeBlock(
  ```python
  def plot_simple_error(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    w_range: NDArray[np.float64],
    x_marker_position: float,
  ) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=(10, 6))
    errors = np.array([cost_function(y, simple_hypothesis(x, w)) for w in w_range])
    ax.plot(w_range, errors, color="blue", label="J(w)")
    ax.axvline(
        x=x_marker_position,
        color="red",
        linestyle="--",
        label=f"w = {x_marker_position}",
    )
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.set_title("Cost as a function of w - J(w)")
    ax.legend()

    return fig, ax
  ```
)

This allow us to visualize the behavior of the cost function by using a *known model* and a range of sampling values for $w$. In the example below, we are using:

$ f_(w) =  (4 x) / 3 $

#codeBlock(
  ```python
  w: float = 4 / 3
  x_train = np.linspace(-5, 5, 100)
  y_train = simple_hypothesis(x_train, w)
  w_sample_range = np.linspace(-5, 8, 100)

  fig, ax = plot_simple_error(
      x=x_train, y=y_train, w_range=w_sample_range, x_marker_position=w
  )
  ```
)

@simple-cost shows the resulting plot. We can observe how the cost approaches a minimum as we change the value of $w$ from both sides, converging to a value close to $1.33$.

#figure(
  image("./images/cost-linear-reg-line.png"),
  caption: [
    Plot of the cost function $J(w)$ as a function of $w$ for the target $f_(w) =  (4 x) / 3$.
  ]
)<simple-cost>

A similar approach can be used to now introduce $b$ as a second target parameter. For example, using a target of the form:

$ f_(w, b) = (2 x) / 5 - 3 / 2  $

#codeBlock(
  ```python
  w = 2.5
  b = -1.5
  x_train = np.linspace(-5, 5, 100)
  y_train = complex_hypothesis(x_train, w, b)
  w_sample_range = np.linspace(-5, 5, 100)
  b_sample_range = np.linspace(-5, 5, 100)

  fig, ax = plot_complex_error_with_contour(
      x=x_train, y=y_train, w_range=w_sample_range, b_range=b_sample_range
  )
  ```
)

@complex-cost shows the resulting plot. We can observe how the cost function has a minimum at $(w, b) = (2.5, -1.5)$ but it is a bit more difficult to observe. As we increase the number of dimensions in the feature space, it becomes even more difficult to visualize the cost function.

But the main idea is that for linear regression, the cost function is *convex and will always have a global minimum.*

#figure(
  image("./images/cost-linear-reg-contour.png"),
  caption: [
    Plot of the cost function $J(w, b)$ as a function of $w$ and $b$ for the target $f_(w, b) = (2 x) / 5 - 3 / 2$.
  ]
)<complex-cost>