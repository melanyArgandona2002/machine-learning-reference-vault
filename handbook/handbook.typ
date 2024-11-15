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
