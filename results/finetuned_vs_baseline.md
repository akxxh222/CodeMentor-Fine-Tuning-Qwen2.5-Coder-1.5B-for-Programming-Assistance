# CodeMentor Adapter V2 Evaluation

## Development-regression outcome

The active V2 adapter achieved the highest combined score on this 11-prompt development-regression sample.

| Model | Correctness (/22) | Relevance (/22) | Completeness (/22) | Code validity (/22) | Total (/88) | Percentage |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Qwen | 18 | 22 | 18 | 16 | 74 | 84.09% |
| Previous adapter V1 | 15 | 22 | 18 | 15 | 70 | 79.55% |
| **Fine-tuned adapter V2** | **18** | **22** | **19** | **19** | **78** | **88.64%** |

Fine-tuned V2 achieved **88.64%**, compared with **84.09%** for baseline Qwen and **79.55%** for adapter V1. That is an improvement of **4.55 percentage points** over the baseline and **9.09 percentage points** over V1. Its correctness score ties the baseline while code validity and completeness exceed it. This is evidence of improvement on this sample, not proof of general superiority.

These 11 prompts were selected from `data/test/test.json` and used for the promotion decision. They must therefore be treated as development regression prompts, not as an unbiased final test. A future final evaluation must use fresh prompts or exclude these 11 prompts from the existing 500-example test set.

## Training evidence

- Training examples: 1,000 (500 Python, 388 JavaScript, 111 Java, 1 Go)
- Validation examples: 100
- Training-set prompt overlap with the original validation/test sets: 0
- Objective: conversational prompt/completion with `completion_only_loss=True`
- Epochs: 2
- Optimizer steps: 250
- Learning rate: `5e-5`
- Best/final validation loss: `0.3921625912`
- Promoted V2 path: `models/adapter`
- Previous adapter rollback path: `models/adapter_v1_backup`

## Dataset-quality limitations

The 1,000-example training set is **syntax/compile-screened, not semantically certified**. Local validators confirm properties such as parsability, balanced code fences, supported language, and successful compilation where a compiler is available. They cannot prove that an answer implements the requested behavior.

An independent audit found incorrect targets that still pass those mechanical checks, including JavaScript prime filtering that treats `1` as prime, a binary-search boundary error, a word-reversal answer that reverses characters instead of words, and inconsistent claimed outputs. The active adapter should therefore be described as an empirically improved V2 candidate, not a correctness-certified model.

Language coverage is also narrow: Python (500), JavaScript (388), Java (111), and Go (1). C, C++, C#, Bash, SQL, and CSS were excluded because the conservative curation pipeline had no configured validator for them. This limits conclusions about multilingual improvement.

## V2 score table

| # | Correctness (0-2) | Relevance (0-2) | Completeness (0-2) | Code validity (0-2) | Total |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 2 | 2 | 8 |
| 2 | 2 | 2 | 2 | 2 | 8 |
| 3 | 2 | 2 | 2 | 2 | 8 |
| 4 | 2 | 2 | 2 | 2 | 8 |
| 5 | 2 | 2 | 2 | 2 | 8 |
| 6 | 2 | 2 | 2 | 2 | 8 |
| 7 | 1 | 2 | 2 | 1 | 6 |
| 8 | 1 | 2 | 1 | 1 | 5 |
| 9 | 2 | 2 | 1 | 1 | 6 |
| 10 | 1 | 2 | 1 | 2 | 6 |
| 11 | 1 | 2 | 2 | 2 | 7 |
| **Total** | **18** | **22** | **19** | **19** | **78/88** |

## Verbatim V2 responses

### 1. Java class

```java
public class MyClass {
    private int myField;

    public MyClass(int myField) {
        this.myField = myField;
    }

    public void myMethod() {
        System.out.println("My Method");
    }
}
```

**Assessment:** Correct, relevant, complete, and valid.

### 2. Oracle SQL average

```sql
SELECT AVG(cost) 
FROM products;
```

**Assessment:** Correct, relevant, complete, and valid under the schema assumed by the prompt.

### 3. Python mean

```python
def mean(data):
    return sum(data) / len(data)

# Example usage:
data = [10, 20, 30, 40]
print(mean(data)) # Output: 25.0
```

**Assessment:** Correct for the non-empty numerical dataset requested.

### 4. CSS stylesheet

```css
body {
  font-family: Arial, sans-serif;
}

h1 {
  color: #333;
  text-align: center;
}

p {
  margin-bottom: 20px;
}

a {
  color: #007bff;
  text-decoration: none;
}
```

**Assessment:** Correct, relevant, complete, and valid.

### 5. C++ sum

```cpp
#include <iostream>

int main() {
    int sum = 0;
    for (int i = 1; i <= 100; ++i) {
        sum += i;
    }
    std::cout << "The sum of the first 100 integers is: " << sum << std::endl;
    return 0;
}
```

**Assessment:** Correct, relevant, complete, and valid.

### 6. Bash random string

```bash
#!/bin/bash

# Generate a random 3 letter string with no duplicates
random_string() {
    # Create an array of all possible letters
    letters=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "s" "t" "u" "v" "w" "x" "y" "z")

    # Initialize an empty string to store the result
    result=""

    # Loop until we have generated 3 unique letters
    while [ ${#result} -lt 3 ]; do
        # Randomly select a letter from the array
        letter=${letters[$((RANDOM % ${#letters[@]}))]}

        # If the letter is not already in the result, add it
        if [[ ! $result == *$letter* ]]; then
            result+=$letter
        fi
    done

    echo "$result"
}

# Call the function and print the result
echo $(random_string)
```

**Assessment:** Correctly terminates with three distinct lowercase letters. This fixes the previous V1 adapter's non-terminating `/dev/urandom | sort` pipeline.

### 7. JavaScript perfect square

```javascript
function isPerfectSquare(n) {
  let root = Math.sqrt(n);
  return root * root === n;
}
```

**Assessment:** Partially correct. Floating-point round trips make this return `true` for some non-squares, including `11`, `14`, and `17`; it should test whether the square root is an integer.

### 8. Greater of two C integers

```c
#include <stdio.h>

int main() {
    int num1, num2;
    
    printf("Enter two numbers: ");
    scanf("%d %d", &num1, &num2);
    
    if (num1 > num2) {
        printf("%d is greater than %d\n", num1, num2);
    } else {
        printf("%d is greater than %d\n", num2, num1);
    }
    
    return 0;
}
```

**Assessment:** Works for unequal inputs but incorrectly describes the second value as greater when both values are equal.

### 9. C# JSON serializer

```csharp
using System;
using Newtonsoft.Json;

public class JsonSerializer
{
    public static string Serialize(object obj)
    {
        return JsonConvert.SerializeObject(obj);
    }
}
```

**Assessment:** The class and serialization logic are correct. It needs the external `Newtonsoft.Json` package reference to compile in a standalone project.

### 10. Java cube series

```java
public class SumOfSeries {
    public static void main(String[] args) {
        int n = 5;
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += Math.pow(i, 3);
        }
        System.out.println("Sum of series: " + sum);
    }
}
```

**Assessment:** The inclusive summation is now correct and the code runs, but `n` is hard-coded rather than accepted as general input.

### 11. C# adjacent duplicates

```csharp
using System;

class Program
{
    static void Main()
    {
        string str = "azxxzy";
        string result = RemoveAdjacentDuplicates(str);
        Console.WriteLine(result); // azxy
    }

    static string RemoveAdjacentDuplicates(string str)
    {
        char[] arr = str.ToCharArray();
        int i = 0;
        for (int j = 1; j < arr.Length; j++)
        {
            if (arr[i] != arr[j])
            {
                arr[++i] = arr[j];
            }
        }
        return new string(arr, 0, i + 1);
    }
}
```

**Assessment:** The program correctly produces `azxzy`, matching the project's held-out reference behavior, and is complete and compilable. The inline comment incorrectly says `azxy`, so the response is only partially correct overall.

## Promotion decision and scope

The candidate outperformed the previous adapter on this development sample: total score increased from 70 to 78 and correctness from 15 to 18. Compared with baseline, total score increased from 74 to 78 and code validity from 16 to 19 while correctness tied at 18.

V2 was promoted to `models/adapter` under a development rule requiring correctness non-regression plus gains in completeness, code validity, and total score. The previous adapter remains at `models/adapter_v1_backup`. Promotion is reversible and provisional because the sample is small, was used for model selection, and does not establish broad semantic correctness.
