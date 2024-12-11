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

1. *Features*:
   - \( X_j \): Represents the \( j \)-th feature (\( j \) from 1 to \( n \)).
   - \( X^{(i)} \): A vector containing all features for the \( i \)-th training example.
   - $\( X^{(i)}_j \)$: Value of the \( j \)-th feature in the \( i \)-th example.
2. *Model Parameters*:
   - \( W \): A vector containing the weights (\( W_1, W_2, ..., W_n \)).
   - \( b \): The intercept, a single number.
3. *Model with Multiple Features*:
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
2. The model can be rewritten using the *dot product*:
  $ \[
   f_{w,b}(X) = W \cdot X + b
   \]$
   - Dot product: \( W \cdot X = W_1X_1 + W_2X_2 + \ldots + W_nX_n \).

*Key Terms*
- *Multiple Linear Regression*:
  - Uses multiple features to predict the target value.
  - Different from *univariate regression* (1 feature) and "multivariate regression" (which refers to something else).
- *Vectorization*:
  - A technique to efficiently implement models with multiple features.

=== Video 2: Vectorization Part 1

Vectorization is a powerful technique that simplifies and accelerates algorithm implementation, particularly in machine learning contexts.

*Key Concepts*
1. *Definition of Vectorization*:
   - The use of vectorized mathematical operations to replace explicit loops in code.
   - Leverages optimized libraries and advanced hardware, such as multi-core CPUs or GPUs.

2. *Example of Vectorization*:
   - Given a parameter vector \( w \) and a feature vector \( x \), calculating a function \( f \) without vectorization requires explicit loops or manual multiplication and addition.
   - With *NumPy*, a Python linear algebra library, the `np.dot(w, x)` method performs the dot product efficiently, combining multiplication and summation.

3. *Benefits of Vectorization*:
   - *Shorter Code:* Reduces multiple lines to a single line, improving readability and maintainability.
   - *Higher Speed:* Vectorized implementations utilize parallel hardware, resulting in significantly shorter execution times.

4. *Non-Vectorized Implementations*:
   - *Basic Code:* Manually writing each operation.
   - *Using Loops:* Slightly better but inefficient for large \( n \) (e.g., \( n = 100,000 \)).

5. *Vectorized Implementation*:
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
- *Performance:* NumPy uses parallel instructions on modern hardware, speeding up calculations.
- *Simplicity:* Simplifies development and debugging.
- *Scalability:* Ideal for large-scale computations.

In summary, vectorization not only reduces development time by minimizing code but also optimizes performance, fully leveraging modern hardware capabilities.

=== Video 3: Vectorization Part 2

*Vectorization* is a key technique in machine learning algorithms that significantly improves both code efficiency and readability.

1. *Definition and Benefits:*
   - *Vectorization*: The process of using parallel mathematical operations instead of iterating through loops.
   - *Advantages*:
     - Shorter, more readable code.
     - Much faster execution by leveraging specialized hardware like CPUs or GPUs.

2. *Implementation Example:*
   - Without vectorization:
     - Using `for` loops to perform calculations one by one.
   - With vectorization:
     - Operations like *dot product* implemented via libraries like NumPy (e.g., `np.dot()`).
     - Calculations are performed in parallel, improving efficiency.

#figure(
  image("./images/w2_s1_v3_vectorization_without_with.png" )
)

3. *Performance Impact:*
   - *Parallel Hardware*:
     - Processes multiple values simultaneously, reducing execution time.
     - Especially useful for large datasets or models with thousands of features.
   - Comparison:
     - Non-vectorized algorithms may take hours.
     - Vectorized implementations complete the same tasks in minutes.

4. *Applications in Machine Learning:*
   - Parameter updates in *multiple linear regression*:
     - Without vectorization: Each weight (`w_j`) updated in a loop.
     - With vectorization: All weights updated simultaneously via mathematical operations.

#figure(
  image("./images/w2_s1_v3_gradiant_descent.png" )
)

5. *Using NumPy:*
   - *NumPy Arrays*:
     - Primary tool for vectorization in Python.
   - Example:
     - `w = w - 0.1 * d` updates all weights at once.
   - Key functions:
     - `np.dot()` for dot product operations.
     - Efficiently handles large datasets.

6. *Practical Recommendations:*
   - Learn and optimize vectorized code with NumPy.
   - Use timing techniques to compare vectorized vs. non-vectorized implementations.

*Conclusion:*
Vectorization is essential for modern machine learning, enabling algorithms to scale efficiently for large datasets.

=== Video 5: Gradient Descent for Multiple Linear Regression ===

*Multiple Linear Regression*
- In this model, parameters \( w_1, w_2, \ldots, w_n \) (weights) and \( b \) (bias) define the relationship between multiple features \( x_1, x_2, \ldots, x_n \) and the target value \( y \).
- The model can be compactly represented in *vector notation*:
  \[
 $ f_{w,b}(x) = w^T x + b$
  \]
  Where:
  - \( w \): vector of weights.
  - \( x \): vector of features.
  - \( w^T x \): dot product between \( w \) and \( x \).

- The cost function \( J(w, b) \) measures the average error between predictions \( $f_{w,b}(x) $\) and true values \( y \). The goal is to minimize \( J \) to find optimal \( w \) and \( b \).

*Gradient Descent for Multiple Linear Regression*
Gradient descent iteratively updates parameters to minimize \( J \):
- *General Rule*:
  \[
  w_j = w_j - \alpha \frac{\partial J}{\partial w_j}
  \]
  \[
  b = b - \alpha \frac{\partial J}{\partial b}
  \]
  Where \( \alpha \) is the *learning rate*, controlling step size.

- For multiple features (\( n > 1 \)):
  - Each weight \( w_j \) is updated using all features \( x_j \) and the error term \( $(f_{w,b}(x) - y)$ \).
  - The bias \( b \) is updated similarly as in univariate regression.

*Vectorized Implementation*
Vectorization enables efficient calculations, avoiding loops. For example:
- The error term can be computed for all examples at once:
  \[
  \text{error} =$ f_{w,b}(X) - Y$
  \]
  Where \( X \) is a matrix of input vectors, and \( Y \) is a vector of target values.

- Updates for \( w \) and \( b \) are performed using matrix operations, speeding up the process.

*Normal Equation Method*
- Alternatively, \( w \) and \( b \) can be directly calculated (no iterations) using a formula from linear algebra called the *normal equation*.
  - Advantage: Exact for linear regression, no iterative optimization needed.
  - Disadvantages:
    - Not applicable to other algorithms like logistic regression or neural networks.
    - Computationally expensive for datasets with many features.

*Key Notes*
- While the normal equation is useful in some cases, gradient descent is preferred for its flexibility and efficiency in large-scale problems.
- Libraries like NumPy simplify these implementations.
- Feature scaling (normalization) and choosing an appropriate \( \alpha \) are critical for model performance.

== Section 3:
=== Video 1: Feature scaling part 1

1. *Introduction to Feature Scaling*:
   - A technique that allows gradient descent to be faster.
   - Analyzes the relationship between the size of the features (entities) and the associated parameters.

2. *Example of House Price Prediction*:
   - *Features*:
     - `x1`: Size of the house (300-2000 square feet).
     - `x2`: Number of bedrooms (0-5 bedrooms).
   - *First set of parameters*:
     - `w1 = 50`, `w2 = 0.1`, `b = 50`.
     - Prediction is far from the actual price ($dollar 500,000$).
   - *Second set of parameters*:
     - `w1 = 0.1`, `w2 = 50`, `b = 50`.
     - Prediction matches the actual price ($dollar 500,000$).

3. *Relationship Between Range of Values and Parameters*:
   - Features with larger ranges (like `x1`) tend to have smaller parameters.
   - Features with smaller ranges (like `x2`) tend to have larger parameters.

4. *Impact on Cost Function and Gradient Descent*:
   - Different scales in features cause elliptical contours in the cost function.
   - Small changes in `w1` have large impacts on the cost due to the higher range of `x1`.
   - Changes in `w2` have less impact because `x2` has a smaller range.

5. *Problems with Gradient Descent on Unscaled Features*:
   - High and thin contours make descent inefficient.
   - Gradient descent may oscillate and take longer to find the global minimum.

6. *Feature Scaling*:
   - Transformation of data so that all features take similar ranges, such as between 0 and 1.
   - Results:
     - Contours become more circular.
     - Gradient descent finds the global minimum faster.

7. *Conclusion*:
   - Scaling features significantly improves the efficiency of gradient descent.
   - Important for cases where features have very different value ranges.

=== Video 2: Feature Scaling Part 2
1. *Motivation for Feature Scaling*:
   - Features often have different value ranges, which can affect the performance of algorithms like gradient descent.
   - Scaling helps features have comparable ranges, accelerating the optimization process.

2. *Feature Scaling Methods*:

   *Max Scaling:*
   - Divides each value by the maximum range of the feature.
   - Example:
     - If \( x_1 \) ranges from 3 to 2000:
       - Scaling: \( x_1^{\text{scaled}} = \frac{x_1}{2000} \), with values ranging from 0.0015 to 1.
     - If \( x_2 \) ranges from 0 to 5:
       - Scaling: \( x_2^{\text{scaled}} = \frac{x_2}{5} \), with values ranging from 0 to 1.

   *Mean Normalization:*
   - Centers the values around 0 and scales them within a fixed range.
   - Formula:
     \( x^{\text{norm}} = \frac{x - \mu}{\text{range}} \)
     where \(\mu\) is the mean and the range is \(\text{max} - \text{min}\).
   - Example:
     - For \( x_1 \) (\(\mu_1 = 600, \text{range} = 2000 - 300\)):
       \( x_1^{\text{norm}} \) ranges from -0.18 to 0.82.
     - For \( x_2 \) (\(\mu_2 = 2.3, \text{range} = 5 - 0\)):
       \( x_2^{\text{norm}} \) ranges from -0.46 to 0.54.

   *Z-Score Normalization:*
   - Uses the mean (\(\mu\)) and standard deviation (\(\sigma\)) to rescale.
   - Formula:
     \( x^{\text{Z}} = \frac{x - \mu}{\sigma} \).
   - Example:
     - For \( x_1 \):
       \(\mu_1 = 600, \sigma_1 = 450 \Rightarrow x_1^{\text{Z}}\) ranges from -0.67 to -3.1.
     - For \( x_2 \):
       \(\mu_2 = 2.3, \sigma_2 = 1.4 \Rightarrow x_2^{\text{Z}}\) ranges from -1.6 to -1.9.

3. *Scaling Cases*:
   - Features with large ranges (e.g., -100 to 100) or very small ones (e.g., -0.001 to 0.001) benefit from scaling.
   - Features with high values (e.g., body temperature ranging from 98.6 to 105 °F) also need rescaling to avoid slowing down gradient descent.

4. *Impact of Scaling*:
   - Facilitates gradient descent by reducing the dependency on scale.
   - It is a recommended practice and rarely harms the model.

5. *Conclusion*:
   - If there is doubt about the need for scaling, it's better to implement it.
   - Scaling features can significantly improve training efficiency.

=== Video 3: Checking Gradient Descent for Convergence
The text explains how to check if gradient descent is converging towards the global minimum of the cost function \( J \). The key steps include:

1. *Plotting a Learning Curve*:
   - Plot the cost \( J \) (vertical axis) against the number of gradient descent iterations (horizontal axis).
   - If gradient descent is working correctly, \( J \) should decrease after each iteration.

2. *Detecting Problems*:
   - If \( J \) increases in any iteration, it may indicate that the learning rate \( \alpha \) is too high or there is an error in the code.

3. *Convergence*:
   - When the curve stabilizes and \( J \) no longer decreases significantly, the algorithm has converged.
   - A threshold \( \epsilon \) can be used to automatically detect convergence: if \( J \) changes less than \( \epsilon \) (e.g., 0.001) between iterations, it is considered convergent.
   - However, the correct threshold can be difficult to choose, so observing the curve directly is preferable.

4. *Variation in Iterations*:
   - The number of iterations needed for convergence can vary widely depending on the application (tens, thousands, or more).

The analysis of the learning curve helps to adjust parameters and detect if gradient descent is not working correctly.

Here is the translation of the text into English in the requested `typsts` format:

=== Video 4: Choosing the Learning Rate
The text discusses how to choose an appropriate learning rate (\( \alpha \)) for gradient descent and how to identify related issues.

*Key points:*

1. *Impact of \( \alpha \):*
   - If \( \alpha \) is too small, the descent will be slow.
   - If \( \alpha \) is too large, the algorithm may not converge and could even cause the cost \( J \) to increase.

2. *Common issues:*
   - A high learning rate may cause the gradient to oscillate around the minimum without converging.
   - A constant increase in \( J \) may indicate implementation errors, such as using the wrong sign when updating parameters.

3. *Debugging tips:*
   - Use a very small \( \alpha \) to check if \( J \) decreases at each iteration.
   - If it doesn't decrease, there is likely an error in the code.

4. *Method for choosing \( \alpha \):*
   - Test a range of values for \( \alpha \), starting at \( 0.001 \) and gradually increasing (e.g., tripling each time: \( 0.003, 0.01, 0.03 \)).
   - Evaluate \( J \) after a few iterations to find a value that decreases \( J \) quickly and steadily.

5. *Final optimization:*
   - Choose the highest \( \alpha \) that allows stable reduction of \( J \).
   - Perform additional tests to find a good balance between speed and stability.

6. *Optional laboratory:*
   - Explore how the learning rate and feature scaling affect training.
   - Adjust \( \alpha \) and analyze its effects on the model.

This approach ensures effective training and optimizes the choice of \( \alpha \) for the model.

=== Video 6: Feature Engineering
The text discusses *feature engineering* and how designing suitable functions can significantly improve the performance of a learning algorithm. *Key points:*

1. *Importance of features:*
   - Selecting or creating appropriate features is key to improving the model's accuracy.

2. *Practical example:*
   - To predict house prices:
     - \( x_1 \): width (front of the plot).
     - \( x_2 \): depth (length of the plot).
     - Initial model: \( f(x) = w_1x_1 + w_2x_2 + b \).

3. *Model improvement:*
   - Introduce a new feature \( x_3 = x_1 \times x_2 \) (area of the plot).
   - Improved model: \( f(x) = w_1x_1 + w_2x_2 + w_3x_3 + b \).
   - This allows the model to decide which feature (width, depth, or area) is most relevant.

4. *Feature engineering:*
   - Involves transforming or combining original features based on knowledge or intuition about the problem.
   - Helps to adjust models not just for linear data but also for more complex patterns (curves, nonlinear functions).

Feature engineering optimizes the model's inputs, enabling more accurate predictions. In the next content, we will explore how to adjust models to nonlinear patterns.

=== Video 7: Polynomial Regression

1. *Limitations of linear regression:*
   - Fitting straight lines to data is not always sufficient to represent complex relationships.

2. *Polynomial regression:*
   - Involves creating new features by raising the original variable (\( x \)) to different powers (e.g., \( x^2, x^3 \)).
   - Example:
     - \( x \): house size.
     - \( x^2 \): size squared.
     - \( x^3 \): size cubed.
   - This approach allows modeling curves that better fit the data.

3. *Importance of feature scaling:*
   - The new features (\( x^2, x^3 \)) may have very wide ranges, which affects algorithms like gradient descent.
   - It is crucial to normalize features so they have comparable values.

4. *Other transformations:*
   - In addition to powers, transformations such as square roots can be used, which produce smooth, increasing curves.

5. *Feature selection:*
   - The choice of features depends on the relationship with the data.
   - Later, methods will be learned to evaluate the performance of different models and functions.

6. *Implementation and practice:*
   - Practice is suggested with laboratories that implement polynomial regression and explore libraries like *Scikit-learn*, widely used in machine learning.

7. *Next steps:*
   - Next week, classification algorithms will be introduced, which allow predicting categories.

This approach combines theory and practice to deepen the design and implementation of more effective models.

= Week 3: Classification
== Section 1: Classification with logistic regression
=== Video 1: Motivations

1. *Classification vs. Linear Regression:*
   - Classification predicts a specific category, while linear regression predicts a continuous value.
   - Examples of classification:
     - Is this email spam? (yes/no)
     - Is this transaction fraudulent? (yes/no)
     - Is this tumor malignant? (yes/no)

2. *Binary Classification:*
   - In binary classification, the output variable can only take two possible values (e.g., yes/no or 0/1).
   - Commonly used terms:
     - *Zero (0):* Negative class (e.g., Not spam).
     - *One (1):* Positive class (e.g., Spam).
   - Other terms include *true (1)* and *false (0)*, or *positive (1)* and *negative (0)*, which help to explain the concept.

3. *Issues with Linear Regression in Classification:*
   - Linear regression is unsuitable for classification because it predicts continuous values, which may not correspond to binary classification.
   - Applying a threshold (e.g., 0.5) might work in some cases but is not always effective.

4. *Example of Classification with Linear Regression:*
   - Imagine classifying tumors as malignant (1) or benign (0) based on tumor size.
   - Using linear regression, a best-fit line may predict values outside the range [0, 1], which is not useful for proper classification.
   - Adding new data points can shift the best-fit line, leading to misclassification of previously well-classified tumors.

5. *Decision Line Problem:*
   - Adding a new data point may alter the decision boundary, causing incorrect predictions. This demonstrates the unsuitability of linear regression for binary classification.

6. *Logistic Regression:*
   - Unlike linear regression, *logistic regression* is specifically designed for binary classification.
   - Its output always lies between 0 and 1, making it better suited for classifying into two categories (e.g., 0 or 1).
   - Despite the name "regression," logistic regression is used for binary classification.

7. *Conclusion:*
   - Linear regression may work in some cases but is unreliable for binary classification problems.
   - Logistic regression is the recommended technique, as it avoids the issues of linear regression and provides more accurate results.

=== Video 3: Motivations

*Notes on Logistic Regression:*

1. *Purpose:*
   - Logistic regression is a widely used classification algorithm.
   - It is suitable for problems where the goal is to classify data into two categories, such as determining whether a tumor is malignant or benign (labels 1 or 0).

2. *Graph and Output of Logistic Regression:*
   - The horizontal axis represents tumor size, and the vertical axis is 0 or 1, reflecting a binary classification problem.
   - Logistic regression fits an S-shaped curve (sigmoid function) to the data, yielding probabilities between 0 and 1.

3. *Sigmoid Function:*
   - The mathematical formula for the sigmoid function is:
     \[
     g(z) = \frac{1}{1 + e^{-z}}
     \]
   - The function outputs values between 0 and 1, modeling probabilities.
   - For large \( z \), \( g(z) \) approaches 1; for small \( z \), \( g(z) \) approaches 0.

4. *Logistic Regression Model:*
   - Logistic regression uses a linear formula \( wx + b \), passed through the sigmoid function:
     \[
     f(x) = g(wx + b)
     \]
   - This produces a value between 0 and 1, interpreted as the probability of belonging to the positive class (1).

5. *Interpreting Results:*
   - The logistic regression output represents the probability that \( y = 1 \) for a given input \( x \).
   - For example, if the model predicts 0.7, the tumor has a 70% probability of being malignant (1).
   - Probabilities must sum to 1: if \( P(y = 1) = 0.7 \), then \( P(y = 0) = 0.3 \).

6. *Notation:*
   - Notation \( f(x) = P(y = 1 | x) \) represents the probability of \( y = 1 \) given \( x \), with \( w \) and \( b \) as model parameters.

7. *Code Implementation:*
   - In optional labs, the sigmoid function can be implemented in code to observe how it improves classification tasks.

8. *Practical Applications:*
   - Logistic regression is used in fields like online advertising to decide which ads to show to users based on data and probabilities.

=== Video 5: Decision Boundary

1. *Computing \( z \):*
   - \( z = w \cdot x + b \), where \( w \) are weights, \( x \) is the feature vector, and \( b \) is the bias.

2. *Applying Sigmoid Function:*
   - The sigmoid function \( g(z) \) is applied to \( z \), resulting in the probability \( f(x) = g(z) \):
     \[
     g(z) = \frac{1}{1 + e^{-z}}
     \]

3. *Binary Prediction:*
   - A threshold (e.g., 0.5) is used:
     - If \( f(x) \geq 0.5 \), predict \( y = 1 \).
     - If \( f(x) < 0.5 \), predict \( y = 0 \).

4. *Decision Boundary:*
   - The decision boundary is where \( P(y = 1) = 0.5 \), or \( z = 0 \), forming a line or surface in the feature space.
   - For example, with \( w_1 = 1 \), \( w_2 = 1 \), \( b = -3 \), the boundary is \( x_1 + x_2 = 3 \).

5. *Complex Boundaries:*
   - Adding polynomial terms can create nonlinear boundaries, such as circles (\( x_1^2 + x_2^2 = 1 \)) or ellipses, enabling the model to handle complex data distributions.

== Section 2: Cost Function for Logistic Regression
=== Video 1: Cost Function for Logistic Regression

1. *Cost Function*:
   - The cost function measures how well a set of parameters fits the training data.
   - In logistic regression, the squared error cost function is not ideal because it produces a non-convex cost function, which can result in local minima, making gradient descent ineffective.

2. *Logistic Regression*:
   - Used for binary classification tasks.
   - The logistic regression model is defined using parameters \( w \) and \( b \).
   - The goal is to choose \( w \) and \( b \) based on the training data.

3. *Squared Error in Logistic Regression*:
   - Squared error is unsuitable because it leads to a non-convex cost function, preventing gradient descent from finding the global minimum.

4. *New Cost Function*:
   - A new cost function ensures convexity, allowing gradient descent to converge to the global minimum.
   - This cost function is defined using a loss function based on the negative logarithm of the logistic regression prediction.

5. *Loss Function*:
   - If \( y = 1 \), the loss is the negative logarithm of \( f(x) \).
   - If \( y = 0 \), the loss is the negative logarithm of \( 1 - f(x) \).
   - The loss function evaluates the performance for a single training example and is used to compute the overall cost function as the average loss across all examples.

6. *Loss Interpretation*:
   - If the model predicts a value close to 1 when \( y = 1 \), the loss is small.
   - If the model predicts a value close to 0 when \( y = 0 \), the loss is also small.
   - The loss increases significantly when predictions diverge from the true labels, especially when \( f(x) \) approaches 0 for \( y = 1 \), or \( f(x) \) approaches 1 for \( y = 0 \).

7. *Advantages of the New Cost Function*:
   - The cost function defined with the new loss function is convex, enabling reliable optimization using gradient descent.

8. *Conclusion*:
   - Squared error cost is inappropriate for logistic regression due to local minima. The logistic loss function ensures a convex cost function, improving gradient descent's effectiveness.

=== Video 3: Simplified Cost Function for Logistic Regression

This video explains how the loss and cost functions for logistic regression can be simplified for easier implementation when using gradient descent to optimize model parameters.

1. *Loss Function*:
   The logistic regression loss function for binary classification can be written as:
   \[
   \text{Loss} = -y \log(f(x)) - (1 - y) \log(1 - f(x))
   \]
   where \( f(x) \) is the model's prediction and \( y \) is the true label (0 or 1).

2. *Simplification*:
   This formula is equivalent to the more complex form of the loss function.
   - When \( y = 1 \), the loss becomes \( -\log(f(x)) \).
   - When \( y = 0 \), the loss becomes \( -\log(1 - f(x)) \).
   The simplified formula avoids handling these two cases separately.

3. *Cost Function*:
   The cost function is the average loss over the training set:
   $\[
   J(theta) = frac{1}{m} sum_{i=1}^{m} [ -y^{(i)} log(f(x^{(i)})) - (1 - y^{(i)}) log(1 - f(x^{(i)})) ]$
   \]
   where \( m \) is the number of training examples.

4. *Statistical Basis*:
   This cost function is based on the principle of *maximum likelihood estimation*, a technique to find the most efficient model parameters.
   - The function's convexity facilitates optimization with gradient descent.

== Section 2: Cost Function for Logistic Regression
=== Video 1: Cost Function for Logistic Regression

1. *Cost Function*:
   - The cost function measures how well a set of parameters fits the training data.
   - In logistic regression, the squared error cost function is not ideal because it produces a non-convex cost function, which can result in local minima, making gradient descent ineffective.

2. *Logistic Regression*:
   - Used for binary classification tasks.
   - The logistic regression model is defined using parameters \( w \) and \( b \).
   - The goal is to choose \( w \) and \( b \) based on the training data.

3. *Squared Error in Logistic Regression*:
   - Squared error is unsuitable because it leads to a non-convex cost function, preventing gradient descent from finding the global minimum.

4. *New Cost Function*:
   - A new cost function ensures convexity, allowing gradient descent to converge to the global minimum.
   - This cost function is defined using a loss function based on the negative logarithm of the logistic regression prediction.

5. *Loss Function*:
   - If \( y = 1 \), the loss is the negative logarithm of \( f(x) \).
   - If \( y = 0 \), the loss is the negative logarithm of \( 1 - f(x) \).
   - The loss function evaluates the performance for a single training example and is used to compute the overall cost function as the average loss across all examples.

6. *Loss Interpretation*:
   - If the model predicts a value close to 1 when \( y = 1 \), the loss is small.
   - If the model predicts a value close to 0 when \( y = 0 \), the loss is also small.
   - The loss increases significantly when predictions diverge from the true labels, especially when \( f(x) \) approaches 0 for \( y = 1 \), or \( f(x) \) approaches 1 for \( y = 0 \).

7. *Advantages of the New Cost Function*:
   - The cost function defined with the new loss function is convex, enabling reliable optimization using gradient descent.

8. *Conclusion*:
   - Squared error cost is inappropriate for logistic regression due to local minima. The logistic loss function ensures a convex cost function, improving gradient descent's effectiveness.

=== Video 3: Simplified Cost Function for Logistic Regression

This video explains how the loss and cost functions for logistic regression can be simplified for easier implementation when using gradient descent to optimize model parameters.

1. *Loss Function*:
   The logistic regression loss function for binary classification can be written as:
   \[
   \text{Loss} = -y \log(f(x)) - (1 - y) \log(1 - f(x))
   \]
   where \( f(x) \) is the model's prediction and \( y \) is the true label (0 or 1).

2. *Simplification*:
   This formula is equivalent to the more complex form of the loss function.
   - When \( y = 1 \), the loss becomes \( -\log(f(x)) \).
   - When \( y = 0 \), the loss becomes \( -\log(1 - f(x)) \).
   The simplified formula avoids handling these two cases separately.

3. *Cost Function*:
   The cost function is the average loss over the training set:
   \[
   $J(theta) = frac{1}{m} sum_{i=1}^{m} [ -y^{(i)} log(f(x^{(i)})) - (1 - y^{(i)}) log(1 - f(x^{(i)}))]$
   \]
   where \( m \) is the number of training examples.

4. *Statistical Basis*:
   This cost function is based on the principle of *maximum likelihood estimation*, a technique to find the most efficient model parameters.
   - The function's convexity facilitates optimization with gradient descent.

== Section 5: Gradient Descent for Logistic Regression
=== Video 1: Gradient Descent Implementation

*Goal: Optimizing Parameters \(w\) and \(b\)*
- The goal is to find the optimal values for the parameters \(w\) (weights) and \(b\) (bias) that minimize the cost function \(J(w, b)\) using the *gradient descent* algorithm.

*Gradient Descent for Logistic Regression*
1. *Cost Function Formula*:
   - The aim is to minimize \(J(w, b)\) by calculating the partial derivatives of the cost function with respect to each parameter.

2. *Parameter Updates*:
   - Gradient descent updates the parameters using the formulas:
     \[
     w_j := w_j - \alpha \frac{\partial J(w, b)}{\partial w_j}
     \]
     \[
     b := b - \alpha \frac{\partial J(w, b)}{\partial b}
     \]
     where:
     - \(\alpha\) is the learning rate,
     - \(\frac{\partial J(w, b)}{\partial w_j}\) and \(\frac{\partial J(w, b)}{\partial b}\) are the partial derivatives of \(J(w, b)\) with respect to \(w_j\) and \(b\).

3. *Partial Derivatives*:
   - The partial derivative of \(J(w, b)\) with respect to \(w_j\) is:
     \[
     $frac{1}{m} sum_{i=1}^{m}( f(x^{(i)}) - y^{(i)} ) x_j^{(i)}$
     \]
   - The partial derivative of \(J(w, b)\) with respect to \(b\) is:
     $\[
     frac{1}{m} sum_{i=1}^{m} ( f(x^{(i)}) - y^{(i)})
     \] $

4. *Vectorized Gradient Descent*:
   - Using a vectorized implementation of gradient descent can significantly speed up computation and improve the algorithm's convergence. However, the video does not go into the details of this approach.

*Logistic Regression vs. Linear Regression*
- Although the parameter update formulas in logistic and linear regression look similar, *they are not the same*.
- The key difference lies in the *activation function*:
  - For linear regression: \( f(x) = wx + b \).
  - For logistic regression: \( f(x) = \frac{1}{1 + e^{-(wx + b)}} \) (sigmoid function).
- This distinction allows logistic regression to handle binary classification tasks, whereas linear regression is used for continuous predictions.

*Feature Scaling*
- *Feature Scaling*: Similar to linear regression, scaling features (e.g., normalizing values to a range between -1 and 1) can significantly accelerate the convergence of gradient descent in logistic regression.

== Section 7: The Problem of Overfitting

=== Video 1: The Problem of Overfitting

1. *Overfitting*:
   - Occurs when a model fits the training data too well, capturing even "noise" and random fluctuations. While the model performs perfectly on the training data, its performance on unseen data is poor.
   - An example of overfitting is using a high-order polynomial (e.g., a fourth-order polynomial) to fit the data. This model might perfectly match the training data but fails to generalize due to high variance.

2. *Underfitting*:
   - Happens when the model is too simple to capture the underlying relationships in the data, resulting in poor performance on both training and unseen data.
   - An example of underfitting is using a linear model to predict house prices based on size when the data shows a more complex quadratic relationship.

3. *High Bias and High Variance*:
   - *High Bias* refers to models that underfit the data (e.g., assuming a linear relationship when it's not).
   - *High Variance* refers to models that overfit the training data, making them highly sensitive to small variations.

4. *Generalization*:
   - The goal of machine learning is to find a model that generalizes well, performing accurately on both training and unseen data. Balancing bias and variance is key to achieving this.

=== Video 2: Addressing Overfitting

1. Collect More Data:
   - One of the most effective ways to address overfitting is by increasing the training dataset size. More examples provide the algorithm with a broader variety, reducing variance and improving generalization.

2. Reduce the Number of Features:
   - If the model has too many features (e.g., house size, number of bedrooms, age, etc.), selecting only the most relevant ones can reduce complexity and overfitting. This is known as *feature selection*. While it might reduce information, removing irrelevant features often enhances the model's generalization.

3. Regularization:
   - Regularization reduces the impact of specific features without removing them entirely. Instead of forcing parameters to zero, it adjusts them to avoid large values, which can lead to overfitting. This approach allows the model to retain all features but ensures that none dominates the model excessively.

*Summary*: To reduce overfitting:
- Increase training data.
- Use only the most relevant features.
- Apply regularization to control parameter magnitudes and prevent overfitting.

These techniques enhance the model's ability to generalize and make accurate predictions on unseen data.

=== Video 4: Cost Function with Regularization

1. *Basic Intuition*:
   - Regularization aims to reduce the value of model parameters (e.g., \( W_3 \), \( W_4 \)) to prevent overfitting. Large parameter values often result in overly complex models that fail to generalize.

2. *Modifying the Cost Function*:
   - To apply regularization, a penalty term is added to the cost function. For example, adding \( 1000 \times W_3^2 + 1000 \times W_4^2 \) forces \( W_3 \) and \( W_4 \) to approach zero, simplifying the model and reducing overfitting.

3. *General Regularization*:
   - Instead of penalizing specific parameters, regularization typically applies to all parameters. A common penalty term is \( $lambda times sum_{j=1}^{n} W_j^2$ \), where \( n \) is the number of features, and \( \lambda \) is the regularization parameter.

4. *Scaling and Convention*:
   - The regularization term is often scaled by \( \frac{\lambda}{2m} \), where \( m \) is the training set size. By convention, the bias term \( b \) is not regularized as it has minimal impact in practice.

5. *Balancing Objectives*:
   - The new cost function balances two goals:
     - Minimize training error.
     - Penalize large parameters.
   - The value of \( \lambda \) determines this balance:
     - \( \lambda = 0 \): Overfitting occurs.
     - Large \( \lambda \): The model becomes overly simplified.
     - Optimal \( \lambda \): Balances error reduction and model simplicity.

6. *Example: Predicting House Prices*:
   - With \( \lambda = 0 \), the model may overfit.
   - With a large \( \lambda \), the model may oversimplify (e.g., fit a straight line).
   - Choosing an appropriate \( \lambda \) allows the model to generalize effectively.

*Conclusion*: Regularization mitigates overfitting by penalizing large parameter values. Adjusting \( \lambda \) helps balance the trade-off between fitting the data and maintaining model simplicity, improving generalization.

Here is the translation of the provided content into English, formatted in Typst:

== Video 5: Regularized Linear Regression
The cost function now includes two components: the traditional quadratic error and an additional regularization term that penalizes large parameter values \(w\). The key difference in gradient descent is that the parameter updates slightly change due to the inclusion of regularization. However, the format remains similar to non-regularized linear regression, with the regularization term affecting only \(w\), not \(b\).

Parameter Updates:
- *For \(w_j\):* The update formula for \(w_j\) includes a regularization term proportional to \(w_j\). This term is multiplied by a value close to 1 (depending on the learning rate \(\alpha\), the regularization parameter \(\lambda\), and the training set size \(m\)). The effect of regularization is to reduce \(w_j\) values at each gradient descent step, helping to prevent overfitting.

- *For \(b\):* The parameter \(b\) is not regularized, so its update remains the same as in non-regularized linear regression.

Derivatives:
The derivatives of the cost function with respect to \(w_j\) and \(b\) account for the regularization term. The derivative with respect to \(w_j\) includes the additional regularization term, whereas the derivative with respect to \(b\) remains unchanged, as in traditional linear regression.

Impact of Regularization:
The regularization term helps shrink the values of \(w_j\), improving model performance, especially when dealing with many parameters and relatively small datasets. This technique is instrumental in preventing overfitting.

Conclusion:
By implementing regularized linear regression, you can improve model performance, control overfitting, and optimize model generalization.

== Video 6: Regularized Logistic Regression
Regularized logistic regression is used to prevent overfitting by adding a regularization term to the cost function. This term penalizes the parameters \(w_1, w_2, ..., w_n\), preventing them from becoming excessively large and ensuring the model does not overfit the training data. The regularized cost function includes a term \(\lambda\) that penalizes the parameter values \(w\), while \(b\) remains unregularized.

The implementation process is similar to that of regularized linear regression. Gradient descent is used to minimize this new cost function, and the parameter update rules for \(w_j\) are applied in a similar manner. As a result, the model becomes more generalizable, avoiding overfitting noisy features in the training data and enhancing its ability to generalize to unseen data. This approach is critical in real-world machine learning applications and a valuable skill for building robust models.
