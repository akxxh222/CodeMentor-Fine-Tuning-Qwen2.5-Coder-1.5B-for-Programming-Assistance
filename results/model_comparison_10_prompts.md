# Baseline vs Fine-Tuned Model Comparison

## Evaluation setup

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Baseline mode: CodeMentor LoRA adapter disabled
- Fine-tuned mode: CodeMentor LoRA adapter enabled
- Inference: local Flask runtime, deterministic decoding (`do_sample=False`)
- Context: every prompt was sent in an independent session; switching models reset all conversation context
- Scoring: each criterion is scored from 0 to 2 using the rubric below

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Correctness | Wrong | Partially correct | Correct |
| Relevance | Doesn't answer | Mostly relevant | Directly relevant |
| Completeness | Major parts missing | Some missing | Complete |
| Code validity | Doesn't work | Minor fix needed | Works |

## Score summary

| # | Base Correctness (0-2) | Base Relevance (0-2) | Base Completeness (0-2) | Base Code Validity (0-2) | FT Correctness (0-2) | FT Relevance (0-2) | FT Completeness (0-2) | FT Code Validity (0-2) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 4 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 5 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 6 | 1 | 2 | 2 | 1 | 0 | 2 | 2 | 0 |
| 7 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 8 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 |
| 9 | 1 | 2 | 1 | 0 | 1 | 2 | 1 | 1 |
| 10 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 1 |
| 11 | 2 | 2 | 1 | 1 | 0 | 2 | 1 | 0 |
| **Total (/88)** | **18** | **22** | **18** | **16** | **15** | **22** | **18** | **15** |
| **Combined score (/88)** | **74** |  |  |  | **70** |  |  |  |

## 1. Java class with a field, constructor, and method

**Question:** How would you create a class in Java and add a field, constructor and method?

### Baseline response

~~~~text
To create a class in Java and add a field, constructor, and method, you can follow these steps:

1. Open your preferred Integrated Development Environment (IDE) such as IntelliJ IDEA or Eclipse.
2. Create a new Java project if it doesn't already exist.
3. Right-click on the package where you want to create the class and select "New" -> "Class".
4. Name the class, for example, `Person`.
5. In the class editor, add the following code:
```java
public class Person {
    // Field declaration
    private String name;
    private int age;

    // Constructor
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // Method declaration
    public void introduce() {
        System.out.println("Hello, my name is " + name + " and I am " + age + " years old.");
    }
}
``
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** The Java code is complete and valid. The response's final Markdown fence is truncated, but that does not alter the Java source.

### Fine-tuned response

~~~~text
public class MyClass {
    private int myField;

    public MyClass(int myField) {
        this.myField = myField;
    }

    public void myMethod() {
        // Method implementation
    }
}
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Direct, complete, and valid Java class containing all requested elements.

## 2. Oracle SQL average product cost

**Question:** Create an Oracle SQL statement for finding the average cost of a product.

### Baseline response

~~~~text
To find the average cost of a product in Oracle SQL, you can use the AVG function on the price column from the products table. Here's an example:

```sql
SELECT AVG(price) AS average_cost
FROM products;
```

This query will return the average price of all products stored in the "products" table. The result will be aliased as "average_cost".
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Correct Oracle-compatible aggregate query. The schema was not supplied, so `products.price` is a reasonable assumed name for product cost.

### Fine-tuned response

~~~~text
SELECT AVG(cost) FROM products;
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Correct and concise Oracle-compatible query using a column name that directly matches the prompt.

## 3. Python numerical mean

**Question:** In Python, how can we calculate the mean of a numerical data set?

### Baseline response

~~~~text
To calculate the mean of a numerical data set in Python, you can use the `numpy` library, which provides a function called `mean()`. Here's an example code snippet:

```python
import numpy as np

# Define a list of numerical data
data = [10, 20, 30, 40, 50]

# Calculate the mean using numpy
mean_value = np.mean(data)

print("The mean of the data is:", mean_value)
```

This code snippet first imports the `numpy` library, which is essential for numerical operations. It then defines a list of numerical values. Using the `np.mean()` function from the `numpy` library, it calculates the mean of these values and stores it in the variable `mean_value`. Finally, it prints out the calculated mean value. This method is efficient and leverages the power of the `numpy` library to handle large datasets and complex
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** The executable example is correct and complete. Only the trailing explanatory sentence is truncated.

### Fine-tuned response

~~~~text
def calculate_mean(data):
    total = 0
    for num in data:
        total += num
    return total / len(data)
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Correct for a non-empty numerical data set and directly answers the question.

## 4. Basic CSS stylesheet

**Question:** Create a CSS stylesheet and include styles for basic elements such as headings, paragraphs, and links.

### Baseline response

~~~~text
Sure! Below is an example of a simple CSS stylesheet that includes styles for headings, paragraphs, and links:

```css
/* Basic Styles */
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
}

h1, h2, h3, h4, h5, h6 {
  color: #333;
  text-align: center;
}

p {
  color: #666;
  line-height: 1.6;
}

a {
  color: #007BFF; /* Blue color for links */
  text-decoration: none;
  transition: color 0.3s ease-in-out;
}

a:hover {
  color: #0056b3; /* Darker blue on hover */
}
```

### Explanation:
- **Body**: Sets the base font family, removes default margins and paddings.
- **Headings
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** The requested stylesheet is complete and valid. The optional explanation is truncated after the complete CSS block.

### Fine-tuned response

~~~~text
body {
    font-family: sans-serif;
}

h1 {
    font-size: 2em;
    color: #333;
}

p {
    font-size: 1.2em;
    line-height: 1.6;
    margin-bottom: 1em;
}

a {
    text-decoration: none;
    color: #007bff;
}
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Valid CSS with representative heading, paragraph, and link styles as requested.

## 5. C++ sum of the first 100 integers

**Question:** Write a C++ program to print out the sum of the first 100 integers.

### Baseline response

~~~~text
Certainly! Below is a simple C++ program that calculates and prints the sum of the first 100 integers.

```cpp
#include <iostream>

int main() {
    int sum = 0;
    for (int i = 1; i <= 100; i++) {
        sum += i;
    }
    std::cout << "The sum of the first 100 integers is: " << sum << std::endl;
    return 0;
}
```

### Explanation:
1. **Include Header**: The `#include <iostream>` header is included to use input/output functions.
2. **Main Function**: The `main()` function is where the execution of the program begins.
3. **Variable Declaration**: An integer variable `sum` is declared to store the result of the summation.
4. **Loop**: A `for` loop is used to iterate from 1 to 100
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Complete, compilable program with the correct inclusive loop. Only the optional explanation is truncated.

### Fine-tuned response

~~~~text
#include <iostream>

int main() {
    int sum = 0;
    for (int i=1; i<=100; i++) {
        sum += i;
    }
    std::cout << "The sum of the first 100 integers is: " << sum << std::endl;
    return 0;
}
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Complete, compilable program with the correct inclusive loop.

## 6. Bash random three-letter string without duplicates

**Question:** Create a bash script that generates a random 3 letter string with no duplicates in it.

### Baseline response

~~~~text
Here is a simple bash script that generates a random 3 letter string without any duplicates:

```bash
#!/bin/bash

# Generate a random 3 letter string
random_string=$(echo {a..z} | fold -w1 | shuf | head -n 3)

# Check if the generated string has duplicates
while [[ $(echo "$random_string" | grep -o . | sort | uniq | wc -l) -ne 3 ]]; do
    # If there are duplicates, generate a new one
    random_string=$(echo {a..z} | fold -w1 | shuf | head -n 3)
done

# Print the generated string
echo $random_string
```

This script uses the `fold` command to split the alphabet into individual characters, the `shuf` command to shuffle them randomly, and the `head` command to select only the first three characters. The `grep
~~~~

**Scores:** Correctness 1 · Relevance 2 · Completeness 2 · Code validity 1

**Assessment:** The approach is relevant and structurally complete, but `echo {a..z} | fold -w1` also produces spaces as candidates. The output is therefore not guaranteed to contain exactly three letters.

### Fine-tuned response

~~~~text
#!/bin/bash

# Generate a random 3 letter string with no duplicates
while true; do
    str=$(cat /dev/urandom | tr -dc 'a-zA-Z' | fold -w 3 | sort -u)
    if [[ $str =~ ^[a-z]{3}$ ]]; then
        break
    fi
done

echo "$str"
~~~~

**Scores:** Correctness 0 · Relevance 2 · Completeness 2 · Code validity 0

**Assessment:** The pipeline feeds an endless `/dev/urandom` stream into `sort`, which waits for end-of-file and never completes. It also generates uppercase letters but accepts only lowercase output.

## 7. JavaScript perfect-square check

**Question:** Create a JavaScript function to check if an integer n is a perfect square.

### Baseline response

~~~~text
Here's a simple JavaScript function that checks if a given integer `n` is a perfect square:

```javascript
function isPerfectSquare(n) {
  // Check if the number is negative or not an integer
  if (n < 0 || !Number.isInteger(n)) {
    return false;
  }

  // Calculate the square root of the number
  const sqrt = Math.sqrt(n);

  // Check if the square root is an integer
  return sqrt === Math.floor(sqrt);
}
```

This function works as follows:

1. It first checks if the input `n` is less than zero or not an integer using the `Number.isInteger()` method. If either condition is true, it returns `false` because a perfect square cannot be negative or non-integer.

2. If the input `n` passes the previous checks, it calculates the square root of `n` using the `Math.sqrt()` function.

3
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Correct, robust implementation. Only the optional explanation is truncated.

### Fine-tuned response

~~~~text
function isPerfectSquare(n) {
  let root = Math.sqrt(n);
  return root === Math.floor(root);
}
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 2 · Code validity 2

**Assessment:** Correct for the integer input required by the prompt, including negative integers, for which it returns `false`.

## 8. C program to find the greater of two integers

**Question:** Design a C code to accept two integers from the user and find the greatest among them.

### Baseline response

~~~~text
Here is a simple C code that accepts two integers from the user and finds the greatest among them:

```c
#include <stdio.h>

int main() {
    int num1, num2;

    printf("Enter first number: ");
    scanf("%d", &num1);

    printf("Enter second number: ");
    scanf("%d", &num2);

    if (num1 > num2) {
        printf("The greater number is %d\n", num1);
    } else {
        printf("The greater number is %d\n", num2);
    }

    return 0;
}
```

In this code, we first include the `stdio.h` header file which provides functions for input/output operations.

Then, we declare two integer variables `num1` and `num2`.

We use the `printf()` function to display a message asking the user to enter the first number, and then use the `scanf()` function
~~~~

**Scores:** Correctness 1 · Relevance 2 · Completeness 1 · Code validity 1

**Assessment:** Works for unequal values, but equal inputs incorrectly report the second number as greater. An equality branch is missing.

### Fine-tuned response

~~~~text
#include <stdio.h>

int main()
{
    int num1, num2;
    
    printf("Enter two numbers: ");
    scanf("%d %d", &num1, &num2);
    
    if (num1 > num2)
        printf("%d is greater than %d\n", num1, num2);
    else
        printf("%d is greater than %d\n", num2, num1);
    
    return 0;
}
~~~~

**Scores:** Correctness 1 · Relevance 2 · Completeness 1 · Code validity 1

**Assessment:** Works for unequal values, but equal inputs incorrectly report the second number as greater. An equality branch is missing.

## 9. C# class for JSON serialization

**Question:** Create a C# class to serialize a JSON object.

### Baseline response

~~~~text
Sure! Below is an example of a C# class that can be used to serialize a JSON object using the `Newtonsoft.Json` library.

First, you need to install the Newtonsoft.Json package via NuGet Package Manager:

```bash
Install-Package Newtonsoft.Json
```

Then, you can create a class called `JsonSerializer` with methods to serialize and deserialize JSON objects.

Here's the code:

```csharp
using System;
using Newtonsoft.Json;

public class JsonSerializer
{
    // Method to serialize an object to JSON
    public static string Serialize(object obj)
    {
        if (obj == null)
        {
            throw new ArgumentNullException(nameof(obj));
        }

        return JsonConvert.SerializeObject(obj);
    }

    // Method to deserialize a JSON string back to an object
    public static T Deserialize<T>(string json)
    {
        if (string.IsNullOrEmpty(json))
        {
            throw new ArgumentException(nameof(json), "JSON string cannot be null or
~~~~

**Scores:** Correctness 1 · Relevance 2 · Completeness 1 · Code validity 0

**Assessment:** The requested serialization method is conceptually correct, but the generated source is truncated inside an unterminated string and lacks the remaining closing braces, so the response as returned does not compile.

### Fine-tuned response

~~~~text
public class JsonSerializer 
{
    public static string Serialize(object obj)
    {
        return JsonConvert.SerializeObject(obj);
    }
}
~~~~

**Scores:** Correctness 1 · Relevance 2 · Completeness 1 · Code validity 1

**Assessment:** The serialization call is correct, but the standalone class omits the `Newtonsoft.Json` import or a fully qualified `JsonConvert` reference and does not state the required package dependency.

## 10. Java sum of cubes through n

**Question:** Write a Java program to calculate the sum of the series 1^3+2^3+3^3+.....+n^3.

### Baseline response

~~~~text
Here is a simple Java program that calculates the sum of the series 1^3 + 2^3 + 3^3 + ... + n^3:

```java
import java.util.Scanner;

public class SumOfSeries {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter the value of n: ");
        int n = scanner.nextInt();
        
        double sum = 0;
        
        for(int i=1; i<n; i++) {
            sum += Math.pow(i, 3);
        }
        
        System.out.println("The sum of the series 1^3 + 2^3 + 3^3 + ... + " + n + "^3 is: " + sum);
    }
}
```

In this program, we first import the `Scanner` class from the `java.util` package which allows us to read input
~~~~

**Scores:** Correctness 1 · Relevance 2 · Completeness 1 · Code validity 1

**Assessment:** The program compiles and runs, but `i < n` omits `n^3`; the required series needs an inclusive upper bound.

### Fine-tuned response

~~~~text
public class SumOfSeries {
    public static void main(String[] args) {
        int n = 5;
        int sum = 0;
        for (int i=1; i<n; i++) {
            sum += Math.pow(i, 3);
        }
        System.out.println("Sum of series: " + sum);
    }
}
~~~~

**Scores:** Correctness 1 · Relevance 2 · Completeness 1 · Code validity 1

**Assessment:** The response uses the right general summation structure, but `i < n` omits `n^3`, and `n` is hard-coded rather than supplied as a general input.

## 11. C# removal of adjacent duplicates

**Question:** Write a C# program to remove all adjacent duplicates from a string. `string str = "azxxzy"`

### Baseline response

~~~~text
Here is a simple C# program that removes all adjacent duplicates from a given string:

csharpCopy
```csharp
using System.Security.Cryptography;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System;

class Program
{
    static void Main()
    {
        string str = "azxxzy";
        Console.WriteLine("Original String: " + str);

        string result = RemoveAdjacentDuplicates(str);
        Console.WriteLine("String after removing adjacent duplicates: " + result);
    }

    static string RemoveAdjacentDuplicates(string str)
    {
        if (string.IsNullOrEmpty(str))
            return str;

        StringBuilder sb = new StringBuilder();
        sb.Append(str[0]);

        for (int i = 1; i < str.Length; i++)
        {
            if (str[i] != sb[sb.Length - 1])
                sb.Append(str[i]);
        }

        return sb.ToString();
    }
```
~~~~

**Scores:** Correctness 2 · Relevance 2 · Completeness 1 · Code validity 1

**Assessment:** The algorithm correctly collapses the adjacent `xx` run to one character, producing `azxzy`, which matches this project's test-set reference behavior. The returned program lacks the final closing brace for `class Program`, so it needs a minor syntax fix before it compiles.

### Fine-tuned response

~~~~text
public static string RemoveAdjacentDuplicates(string str) {     StringBuilder sb = new StringBuilder();     for (int i = 0; i < str.Length - 1; i++)     {         if (str[i] != str[i + 1])             sb.Append(str[i]);     }     return sb.ToString(); }
~~~~

**Scores:** Correctness 0 · Relevance 2 · Completeness 1 · Code validity 0

**Assessment:** For `azxxzy`, this method returns `azxz`: it retains one of the duplicate `x` characters and always omits the final character. It is only a method rather than the requested complete program and does not produce the required result.

## Overall result

- Baseline: **74/88 (84.09%)**
- Fine-tuned: **70/88 (79.55%)**
- Difference: baseline leads by **4 points** on this 11-prompt sample.

Both modes performed strongly on straightforward code generation. Shared weaknesses include equal-input handling in question 8 and the exclusive upper bound in question 10. The fine-tuned model's largest failures include the non-terminating Bash pipeline in question 6 and its incorrect duplicate-removal method in question 11. The baseline response for question 11 uses the correct algorithm but needs a closing class brace, while its question 9 C# response is truncated and non-compilable.
