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

= Class 2: Advanced Learning Algorithms

== Week 1: Neural Networks

=== Section 1: Neural Networks Intuition

==== Video 2: Neurons and the Brain

*Origins and Biological Inspiration*
- Artificial neural networks were inspired by the human brain.
- Initially based on simplified models of biological neurons:
  - *Biological neurons*: Receive electrical impulses through dendrites, perform calculations, and send signals via axons.
  - *Artificial neurons*: Take numerical inputs, perform mathematical calculations, and produce outputs.
- Today, neural networks have little resemblance to the functioning of the human brain.

*History and Resurgence*

- Initial work: 1950s; gained popularity in the 80s and 90s for tasks like handwritten digit recognition.
- Fell out of favor in the late 90s.
- Resurgence since 2005, associated with deep learning, a trending term.
- Neural networks have since revolutionized fields such as:
  - Speech recognition.
  - Computer vision (e.g., the "ImageNet moment" in 2012).
  - Natural language processing (NLP).
  - Applications in medical imaging, climate change, advertising, and more.

#figure(
  image("./images/C2_W1_S1_V2_Neural_Networks.png" )
)

*Key Factors Behind the Resurgence*

1. *Availability of Digital Data*:
   - Exponential increase in data due to digitalization.
   - Digital records now include medical, transactional, and digital interaction data.
2. *Limitations of Traditional Algorithms*:
   - Algorithms like logistic regression did not scale well with large data volumes.
3. *Advantages of Large Neural Networks*:
   - Performance scales with more data and neurons.
   - Effectively leverage "big data" for solving complex problems.
4. *Advancements in Hardware*:
   - GPUs optimized for deep learning, originally designed for graphics.

#figure(
  image("./images/C2_W1_S1_V2_Neural_Brains.png" )
)

 *Notes*
- While neural networks began with a biological motivation, they now focus on engineering principles.
- There is still much to learn about the human brain, but simplified neuron models are highly effective in practice.

==== Video 3: Demand Prediction

*1. Initial Example: Demand Prediction*
- *Context:* Predicting whether a T-shirt will be a bestseller.
- *Data:* Features such as price, shipping costs, marketing investments, and material quality.
- *Practical Application:* Helps plan inventory and marketing campaigns for retailers.

#figure(
  image("./images/C2_W1_S1_V3_Demand_Prediction.png" )
)

*2. Concept of an Artificial Neuron*
- *Simplified Model:*
  - The artificial neuron is a simplified model of a biological neuron.
  - Takes an input (e.g., price) and calculates an output using a formula, representing a probability.
- *Terminology:*
  - "A" or *activation*: Neuron output, representing energy or probability.
- *Biological Relation:* Although much simpler, artificial neurons are effective in practice.

*3. Building a Neural Network*
- *Layers:*
  - A group of connected neurons.
  - *Input Layer:* Receives original features (price, shipping, marketing, material quality).
  - *Hidden Layer:* Processes data to generate intermediate activations (affordability, awareness, perceived quality).
  - *Output Layer:* Predicts the probability of success.
- *Fully Connected:* Each neuron in a layer receives inputs from all outputs of the previous layer.

#figure(
  image("./images/C2_W1_S1_V3_Layers.png" )
)

*4. Detailed Example: Complex Prediction*
- *Three Subproblems:*
  - Estimating *affordability* based on price and shipping.
  - Determining *awareness* using marketing investments.
  - Evaluating *perceived quality* based on price and material quality.
- *Integration:* These estimations are processed by a final neuron to predict success.

#figure(
  image("./images/C2_W1_S1_V3_Layers_multiple.png" )
)

*5. Simplification and Generalization*
- *Vector Representation:*
  - Inputs and outputs are grouped into vectors for implementation simplicity.
- *Automated Calculations:*
  - The network learns to prioritize relevant features without manual intervention.

#figure(
  image("./images/C2_W1_S1_V3_simplify_general.png" )
)

*6. Key Terminology*
- *Input Layer:* Receives original data.
- *Hidden Layer:* Processes intermediate data (activations).
- *Output Layer:* Produces the final result.
- *Activations:* Intermediate values representing the network's internal knowledge.

#figure(
  image("./images/C2_W1_S1_V3_multiple_hidden_layers.png" )
)

*7. Advantages of Neural Networks*
- Transform initial features into more useful representations for prediction.
- Learn complex relationships automatically.

==== Video 4: Example: Recognizing Images
*1. Image Representation*
- Images are represented as *matrices* of pixel intensity values.
- For example, a 1000x1000 pixel image generates a matrix of *1,000,000 values*(brightness intensity between 0 and 255).

#figure(
  image("./images/C2_W1_S1_V4_face_recognition.png" )
)

*2. Neural Network Structure*
- The network takes a *vector* of pixel intensity values as input.
- Includes multiple layers:
  - *Hidden Layers:* Extract intermediate features.
  - *Output Layer:* Predicts the identity of the person in the image.

*3. Features Learned by Layers*
- *First Hidden Layer:* Detects edges and simple lines.
- *Second Hidden Layer:* Combines edges and lines to identify facial parts (eyes, nose, ears).
- *Third Hidden Layer:* Assembles facial parts to recognize complete face shapes.
- *Remark:* The network learns these features from data without explicit instruction.

#figure(
  image("./images/C2_W1_S1_V4_name_layers.png" )
)

*4. Generalization to Other Datasets*
- With different data, the network learns specific features for other objects.
  - Example: Trained on car images, it detects edges, car parts, and complete vehicle shapes.

#figure(
  image("./images/C2_W1_S1_V4_car_classification.png" )
)

*5. Importance and Applications*
- Applicable to various computer vision tasks, including:
  - *Facial recognition.*
  - *Object detection.*
  - *Pattern recognition (e.g., handwritten digits).*


=== Section 3: Neural Network Model

==== Video 1: Neural Network Layer

1. *Neuron Layer*:
   - Fundamental component of a modern neural network.
   - Each layer receives inputs, performs calculations, and generates outputs passed to the next layer.

2. *Neural Network Structure*:
   - Example: A network with 4 input entities, one hidden layer with 3 neurons, and one output layer with a single neuron.
   - Layers are numbered sequentially (input layer: 0, hidden layer: 1, output layer: 2).

3. *Calculation in a Neuron*:
   - Each neuron has parameters \( w \) (weights) and \( b \) (bias).
   - Formula: \( $z = w \c x + b$ \), where \( x \) is the input vector.
   - Activation \( a \) is computed using a sigmoid function: \( a = g(z) \), where \( $g(z) = \f{1}{1 + e^{-z}}$ \).

4. *Notation for Layers and Parameters*:
   - Superscripts (e.g., \[1\], \[2\]) indicate the corresponding layer.
   - Example: \( w^{[1]} \) and \( b^{[1]} \) represent parameters of layer 1.

5. *Hidden Layer Calculations*:
   - Each neuron calculates \( $a_i = g(w_i \c x + b_i)$ \).
   - Produces an activation vector \( $a^{[1]}$ \), used as input for the next layer.

#figure(
  image("./images/C2_W1_S3_V1_neural_network_layer.png" )
)

6. *Output Layer Calculations*:
   - Receives the activation vector \( $a^{[1]}$ \) from the hidden layer.
   - For a single neuron, produces a scalar \( $a^{[2]} = g(w \c a^{[1]} + b)$ \), representing a probability (e.g., \( 0.84 \)).

#figure(
  image("./images/C2_W1_S3_V1_end_layer.png" )
)

7. *Final Prediction*:
   - In binary classification, apply a threshold (e.g., \( 0.5 \)):
     - If \( $a^{[2]} \g 0.5$ \): prediction = 1.
     - If \( $a^{[2]} < 0.5$ \): prediction = 0.

#figure(
  image("./images/C2_W1_S3_V1_final_prediction.png" )
)

8. *Building Larger Networks*:
   - Networks are expanded by chaining multiple hidden layers.
   - Each layer processes outputs of the previous layer, enabling more complex and precise models.

9. *Practical Application*:
   - This approach supports iterative computations from inputs to outputs in models requiring complex predictions.

==== Video 2: More Complex Neural Networks

1. *Neural Network Structure*:
   - Composed of layers: input layer (layer 0), hidden layers (e.g., 1, 2, 3), and output layer.
   - The input layer is typically not counted in the total number of layers.

#figure(
  image("./images/C2_W1_S3_V2_neural_network_structure.png" )
)

2. *Hidden Layer Calculation (Example: Layer 3)*:
   - Receives an activation vector \( a^{[2]} \) from the previous layer.
   - Produces a new activation vector \( a^{[3]} \).
   - Each neuron computes:
     \[
     $a_j^{[3]} = g\l (w_j^{[3]} \c a^{[2]} + b_j^{[3]}\r)$
     \]
     Where:
     - \( $w_j^{[3]}$ \): weights of neuron \( j \) in layer 3.
     - \( $b_j^{[3]}$ \): bias of neuron \( j \) in layer 3.
     - \( g \): activation function (e.g., sigmoid function).

#figure(
  image("./images/C2_W1_S3_V2_more_complex_neural.png" )
)

3. *Notation*:
   - Superscripts (\([l]\)) indicate the layer.
   - Subscripts (\(j\)) indicate the neuron within a layer.
   - Input data \( X \) is often denoted as \( $a_0$ \).

#figure(
  image("./images/C2_W1_S3_V2_notation.png" )
)

4. *Activation Function*:
   - The function \( g \), known as the activation function, determines activation values.
   - Example: sigmoid function. Other functions will be explored later.

5. *Inference Algorithm*:
   - The network predicts values by sequentially calculating activations layer-by-layer using parameters \( w \) and \( b \) and activations from the previous layer.

Here’s the translated text for Section 5 in English:

=== Section 5: TensorFlow Implementation
==== Video 1: Inference in Code

1. *Introduction to TensorFlow*
   - TensorFlow is one of the leading frameworks for implementing deep learning algorithms, often used in projects.

2. *Illustrative Example: Coffee Roasting*
   - A coffee roasting example is used to illustrate how a neural network makes inferences.
   - The controlled parameters are:
     - *Temperature*: The degrees Celsius to which the beans are heated.
     - *Duration*: The time the beans are roasted.
   - A good roast is represented with binary labels:
     - *1*: Good tasting coffee.
     - *0*: Bad tasting coffee.

Image: C01_W02_S05_V01_Coffe_example.png

3. *Dataset Features*
   - Low temperature or insufficient time: Undercooked beans.
   - High temperature or excessive time: Burnt beans.
   - Only certain combinations of temperature and duration produce good coffee.

#figure(
  image("./images/C01_W02_S05_V01_Coffe_build_model.png" )
)

4. *Inference in a Neural Network with TensorFlow*
   - *Input*: Vector `x` with temperature and duration (e.g., `[200, 17]` for 200°C and 17 minutes).
   - *Layer 1*:
     - Type: Dense (3 hidden units).
     - Activation: Sigmoid function.
     - Output: Activation `a1` (e.g., `[0.2, 0.7, 0.3]`).
   - *Layer 2*:
     - Type: Dense (1 unit).
     - Activation: Sigmoid function.
     - Output: Activation `a2` (e.g., `0.8`).
   - *Prediction (`ŷ`)*:
     - Threshold: `0.5`.
     - If `a2 ≥ 0.5`, the prediction is `1` (positive roast). If not, it's `0` (negative roast).

5. *Key Steps for Inference*
   - Create the layers with the necessary specifications (type and activation function).
   - Apply forward propagation:
     - Calculate activations (`a1`, `a2`, etc.).
   - Compare the final result with a threshold to make the prediction.

6. *dditional Details*
   - Use of the TensorFlow library to load parameters (`w` and `b`).
   - Practical lab examples to explore these details.

#figure(
  image("./images/C01_W02_S05_V01_model_for_digit.png" )
)

7. *Additional Example: Handwritten Digit Classification*
   - Input `x`: List of pixel intensity values (numeric matrix).
   - Neural Network:
     - *Layer 1*: Dense (25 units, sigmoid function).
     - *Layer 2*: Dense (10 units, sigmoid function).
     - *Layer 3*: Dense (1 unit, sigmoid function).
   - Inference similar to the coffee example:
     - Forward propagation through multiple layers.
     - Compare the final output with the threshold.

8. *Matrix Structure in TensorFlow*
   - TensorFlow handles data as numerical matrices (tensors).
   - It is essential to understand how data is structured and processed in TensorFlow.

==== Video 2: Data in TensorFlow

*Main Topic*
Representation of data in *NumPy* and *TensorFlow* to implement neural networks within a coherent framework.

*Key Points*
 *Introduction*
- *NumPy*: The standard library for linear algebra in Python, created years ago.
- *TensorFlow*: Created by Google Brain, designed to handle large datasets.
- Differences in representation conventions between *NumPy* and *TensorFlow*.

#figure(
  image("./images/C01_W02_S05_V02_feature_vectors.png" )
)

*Data Representation in TensorFlow and NumPy*
 *Matrices in NumPy*
1. *Matrix Definition and Example*:
   - *Dimensions*: Number of rows x number of columns.
   - Example:
     - `2 x 3`: Two rows, three columns.
     - Code: `x = np.array([[1, 2, 3], [4, 5, 6]])`.

2. *Matrices with Different Dimensions*:
   - `4 x 2`: Four rows, two columns.
   - Code: `x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])`.

3. *Row and Column Vectors*:
   - *1 x 2*: One row, two columns.
     - Example: `x = np.array([[200, 17]])`.
   - *2 x 1*: Two rows, one column.
     - Example: `x = np.array([[200], [17]])`.

4. *1D Vectors*:
   - Simplified representation (without rows or columns): `x = np.array([200, 17])`.
   - Contrast with 2D matrices.

*Matrices and Tensors in TensorFlow*
1. *Tensors*:
   - The internal data representation in TensorFlow for computational efficiency.
   - Example: A `1 x 3` tensor with values `[0.2, 0.7, 0.3]`.
     - Code: `tf.constant([[0.2, 0.7, 0.3]])`.

2. *Conversion Between TensorFlow and NumPy*:
   - From tensor to NumPy array: `a1.numpy()`.
   - Example: `a2.numpy()` converts a `1 x 1` tensor into a NumPy array.

*Code Examples*
1. *Layer 1: Activations*:
   - Code: `a1 = layer_1(x)`.
   - Result: `1 x 3` tensor.

2. *Layer 2: Activations*:
   - Code: `a2 = layer_2(a1)`.
   - Result: `1 x 1` tensor.

*Final Reflection*
- TensorFlow automatically converts NumPy matrices to tensors.
- Conversions between tensors and matrices are common but add complexity.
- Despite the differences, both libraries work well together.

==== Video 3: Building a neural network
1. *Manual Network Building*:
   - In previous methods, layers were created and connected manually, passing input data through each layer.

2. *Using the Sequential Model*:
   - TensorFlow provides the sequential model to simplify the process.
   - This method automatically connects layers to form a neural network.
   - Example:

     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Dense

     model = Sequential([
         Dense(3, activation='sigmoid'),
         Dense(1, activation='sigmoid')
     ])
     ```

3. *Training the Network*:
   - Training data is structured as matrices (X for features and Y for labels).
   - Training:
     ```python
     model.compile(optimizer='adam', loss='binary_crossentropy')
     model.fit(X, Y)
     ```

4. *Predictions with the Model*:
   - Use the trained model for inference:
     ```python
     predictions = model.predict(X_new)
     ```

5. *Common Conventions*:
   - Instead of assigning layers to separate variables, they are typically defined directly within the sequential model.
   - This results in more compact and readable code.

6. *Example Implementations*:
   - Coffee example: Classifying data using a simple two-layer dense model.
   - Digit classification example: Similar to the coffee example, but with multiple layers.

7. *Importance of Understanding the Basics*:
   - Although libraries like TensorFlow simplify development, it is crucial to understand how the underlying algorithms work.
   - You will learn to implement forward propagation from scratch in Python for a deeper understanding.

8. *Balancing Efficiency and Knowledge*:
   - Modern libraries allow you to create advanced neural networks with just a few lines of code.
   - However, it is essential to know the details to diagnose problems and understand what happens behind the scenes.

=== Section 7: Neural network implementation in Python

==== Video 1: Forward prop in a single layer

1. *Activation calculation of the first layer*:
   - Start with an input vector `x` and implement forward propagation to obtain the activation `a2` in a simple neural network.
   - Subscript notation is used to indicate different parameter values (e.g., `w2_1` and `b2_1`).
   - The first value to calculate is the activation `a1_1`, which is obtained by calculating `z1_1` as the dot product between `w1_1` and `x`, adding the bias `b1_1`, and applying the sigmoid function.

2. *Calculation of other activations of the first layer*:
   - Repeat the process to calculate `a1_2` and `a1_3`, using parameters `w1_2` and `b1_2`, and then applying the sigmoid function.
   - The results of the three activations (`a1_1`, `a1_2`, `a1_3`) are grouped into a matrix `a1`.

3. *Forward propagation of the second layer*:
   - The output `a2` is calculated using the parameters of the second layer, `w2_1` and `b2_1`.
   - Calculate `z2_1` as the dot product between `w2_1` and `a1`, add the bias `b2_1`, and then apply the sigmoid function to obtain `a2_1`.

4. *Implementation with NumPy*:
   - The entire forward propagation process is implemented using Python and NumPy (`np`).

*Video 2: General implementation of forward propagation*

1. *Implementation of the dense layer*:
   - Introduce the `dense` function, which takes the activation of the previous layer (`a`), and the parameters of the current layer (`w` and `b`) as inputs.
   - In the example, we work with three neurons in layer 1. The parameters `w_1, w_2, w_3` are stacked into a 2x3 matrix, where each column represents the weights corresponding to each neuron.
   - Similarly, the bias terms (`b`) are stacked into a 1D matrix.

2. *Activation calculation*:
   - The `dense` function generates activations for the current layer. It uses a loop to calculate the activation value `a` for each unit (neuron) using the standard formula: \( z = W \cdot a + b \), followed by the sigmoid function \( a = g(z) \).
   - This process is repeated for each neuron in the layer.

3. *Code for the `dense` function*:
   - The code shows how to extract columns from the matrix `W` and how to use the dot product with the activation from the previous layer. Then, biases are added and the sigmoid function is applied to obtain the activation.

4. *Layer composition*:
   - Explanation of how to combine multiple dense layers sequentially to implement a forward propagation system in a neural network.
   - From the input features `x`, the activation of the first layer \( a_1 \) is calculated, then the second layer \( a_2 \), and so on until the final output \( a_4 \) in a network with four layers.
   - Standard algebraic notation is used: matrices `W` are denoted with uppercase letters and vectors with lowercase letters.

5. *Importance of understanding the low-level process*:
   - Although libraries like TensorFlow or PyTorch are commonly used, it is crucial to understand how they work internally to debug errors and improve performance.
   - Understanding how to implement forward propagation from scratch helps identify issues when things don't work as expected, allowing developers to fix bugs more quickly.



*Section 9: Speculations on artificial general intelligence (AGI)*

*Video 1: Is there a path to AGI?*

1. *Distinction between ANI and AGI*:
   - *ANI (Artificial Narrow Intelligence)*:
     - Refers to systems that perform specific tasks with great efficiency (e.g., virtual assistants, autonomous cars, search engines, applications in agriculture, factories, etc.).
     - Advances in ANI have generated significant value in today's society.
   - *AGI (Artificial General Intelligence)*:
     - The goal is to create AI systems that can perform any task that a typical human can do.
     - While ANI has made great strides, AGI has not made significant progress in practice.

2. *The myth of AI progress towards AGI*:
   - The relationship between ANI progress and AGI is not direct. Progress in ANI does not automatically imply advances in AGI.
   - The rise of modern deep learning has led some to believe that simulating many neurons could replicate human intelligence, but this idea has proven to be overly simplistic.

3. *Difficulties in attempting to simulate the human brain*:
   - *Simplicity of artificial neural networks*: Artificial neural networks are much simpler than biological neurons and do not accurately simulate the behavior of the human brain.
   - *Lack of understanding of the brain*: We still do not fully understand how the brain works, which makes simulating it in a machine an extremely difficult task.

4. *Experiments on brain plasticity*:
   - *Experiments showing the adaptability of the brain*:
     - Animal experiments show that areas of the brain, such as the auditory cortex or somatosensory cortex, can learn to process different types of data (e.g., images instead of sounds).
     - The brain is highly adaptable, suggesting that there could be basic learning algorithms responsible for this plasticity.
     - Examples include using sensors or devices (e.g., cameras mounted on the forehead) that teach people to "see" with different parts of the body, such as the tongue.
     - *Human echolocation*: Echolocation experiments have been conducted where humans learn to use sound to "see" in a manner similar to bats or dolphins.
     - *Haptic belt*: Research has shown that it is possible to teach humans to "feel" direction using devices such as vibrating belts.
   - The brain's plasticity suggests that a single learning algorithm might be able to process different types of sensory inputs.

5. *Hope of replicating the brain with algorithms*:
   - If the brain can learn to perform complex tasks (e.g., seeing, hearing, feeling) with different types of data, perhaps there is an underlying algorithm that we can discover and implement in a machine.
   - Despite progress, the narrator is doubtful that we will ever discover that algorithm and truly replicate the human brain.

6. *Long-term perspective on AGI*:
   - Working on AGI remains one of the most fascinating problems in science and engineering.
   - While there is hope that a path to AGI will one day be discovered, the narrator emphasizes that progress will be extremely difficult.
   - It should not be exaggerated that AGI advancements are near, as we do not fully understand how the human brain works or how to replicate it.

7. *The value of machine learning and neural networks*:
   - While AGI is still a distant goal, neural networks and machine learning are already powerful tools in current applications.

=== Section 10: Vectorization (optional)

==== Video 1: How neural networks are implemented efficiently
*Notes on the vectorized implementation of neural networks:*

1. *Reason for the scalability of neural networks:*
   - Neural networks can be vectorized, allowing for efficient implementation through matrix multiplications.
   - Parallel computing hardware, such as GPUs and certain CPU functions, excels at performing large matrix multiplications, which has been key to the success and scalability of deep learning.

2. *Implementation of Forward Propagation in a single layer:*
   - The code shown is a basic implementation of forward propagation in a single layer of a neural network, where:
     - *X* is the input,
     - *W* are the neuron weights,
     - *B* is the bias.
   - This implementation generates three output numbers by applying the neural network formula.

3. *Vectorized implementation:*
   - Instead of using a `for` loop to process each neuron separately, the process is vectorized, allowing operations to be performed in parallel.
   - *X* is defined as a 2D matrix, just like *W* (the weights), and *B* is also converted into a 2D matrix of size 1x3.
   - Matrix multiplication is performed using `np.matmul`, which is how NumPy performs matrix multiplication.
   - The simplified code is reduced to a few lines:
     - *Z* is calculated as the product of the matrices *X* and *W*.
     - *B* is added to *Z*.
     - Then, the activation function *g* (sigmoid) is applied to *Z* to obtain *A*, which is the output of the layer.

4. *Advantages of the vectorized implementation:*
   - The vectorized implementation is more efficient and faster as it avoids iterative loops and allows parallel computation of operations.
   - The input and output, as well as the parameters *W* and *B*, are now 2D matrices, making forward propagation in a single layer easier.

This vectorized approach is crucial for neural networks to handle large amounts of data and scale efficiently.

==== Video 2: Matrix multiplication

1. *Matrices and Dot Product:*
   - A matrix is a block or 2D array of numbers.
   - The dot product between two vectors is calculated by multiplying corresponding elements and summing the results. Example with vectors (1, 2) and (3, 4): \(1 \times 3 + 2 \times 4 = 11\).

2. *Dot Product and Transposition:*
   - The dot product between two vectors can also be written as the transposition of one vector multiplied by another vector. The transposition of a vector changes its orientation from column to row.

3. *Multiplying a Vector by a Matrix:*
   - A vector can be multiplied by a matrix, and the result is a new matrix.
   - Example: The vector (1, 2) multiplied by the matrix \( \begin{pmatrix} 3 & 4 \\ 5 & 6 \end{pmatrix} \) results in the vector \( (11, 17) \), calculated by performing the dot product between the transposed vector and the columns of the matrix.

4. *Matrix Multiplication:*
   - Matrix multiplication involves dot products between vectors, but organized in a structured way to form the elements of the resulting matrix.
   - Example: If A is a matrix with columns \( a_1 \) and \( a_2 \), and W is a matrix, matrix multiplication is calculated by multiplying the rows of \( A^T \) (the transposition of A) with the columns of W.

5. *Matrix Transposition:*
   - To transpose a matrix, its rows and columns are swapped. This changes the columns of A into the rows of \( A^T \).
   - Example: The matrix \( A = \begin{pmatrix} 1 & 2 \\ -1 & -2 \end{pmatrix} \) becomes \( A^T = \begin{pmatrix} 1 & -1 \\ 2 & -2 \end{pmatrix} \).

6. *Multiplying \( A^T \) and W:*
   - The multiplication of \( A^T \) by W is performed in the same way as the dot product, but with the rows of \( A^T \) and the columns of W.
   - Results from the example: The first element is \( 1 \times 3 + 2 \times 4 = 11 \), the second is \( 1 \times 5 + 2 \times 6 = 17 \), the third is \( -1 \times 3 + -2 \times 4 = -11 \), and the last is \( -1 \times 5 + -2 \times 6 = -17 \).

7. *Generalization:*
   - Matrix multiplication is a combination of dot products between vectors, organized in a specific way to construct the resulting matrix.



==== Video 3: Matrix multiplication rules

1. *Matrix Multiplication:*
   - Matrix A is of size 2x3 (two rows and three columns).
   - The transposition of matrix A is applied, turning its columns into rows.
   - Matrix W has four columns and is considered as vectors w1, w2, w3, w4.

2. *Calculating the Product between Matrices:*
   - To calculate the multiplication between the transposition of A and W, the dot product is performed between the rows of the transposition of A and the columns of W.
   - Each element of the resulting matrix Z is obtained by multiplying a row of the transposed A by a column of W.

3. *Examples of Calculations:*
   - Specific examples are given on how to calculate values in the matrix Z using dot products, such as the value in row 1, column 1 of Z being the result of the dot product between the first row of A transposed and the first column of W.
   - Other examples are analyzed, like the calculation for row 3, column 2, and row 2, column 3.

4. *Matrix Multiplication Requirements:*
   - Matrix multiplication is only valid if the number of columns of the first matrix (A transposed) is equal to the number of rows of the second matrix (W).
   - The dimensions of the resulting matrix Z will be the rows of the first transposed matrix (3) and the columns of the second matrix (4).

5. *Matrix Multiplication Properties:*
   - The matrix product results in a matrix of dimensions determined by the rows of the transposed matrix and the columns of the multiplying matrix.

6. *Application in Vectorized Neural Networks:*
   - The vectorized implementation of neural networks is based on these matrix multiplication principles.
   - Vectorization greatly improves the execution speed of neural networks.

==== Video 4: Matrix multiplication code

1. *Initial Concepts:*
   - Matrix multiplication is analyzed, specifically the transposition of matrix A and its multiplication by matrix W to obtain Z.
   - In code, the transposition of A is performed with `A.T` or `A.transpose()`, and matrix multiplication with `np.matmul(AT, W)`.

2. *Calculations with Matrices:*
   - Matrix W is formed with the weights `w_1`, `w_2`, `w_3`, and matrix B with the biases `b_1`, `b_2`, `b_3`.
   - The calculation of Z as `Z = A.T @ W + B` gives the results 165, -531, and 900, corresponding to the weights of the feature inputs.

3. *Sigmoid Function:*
   - The sigmoid function is applied to the values of Z to obtain the output values, A, which are [1, 0, 1] after rounding.

4. *Implementation in Code:*
   - The transposition of A and multiplication with W and B are implemented as `Z = np.matmul(A.T, W) + B`.
   - The final output of the layer is obtained by applying the activation function `g` to Z: `a_out = g(Z)`.

5. *TensorFlow Conventions:*
   - In TensorFlow, it is common for individual examples to be in rows of matrix X, rather than transposing X.
   - This implementation variation is mentioned as a convention, but both forms are correct.

6. *Advantages of Vectorized Implementation:*
   - Using vectorized operations and efficient matrix multiplication (`matmul`) allows for neural network inference with fewer lines of code and takes advantage of the efficiency of modern computers.

