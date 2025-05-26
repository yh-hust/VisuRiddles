prompt_choice_direct = """
Please select the correct answer based on the question and the provided image, and output the answer in the format <answer>X</answer> (X should be A, B, C, or D). No need to provide any explanations, just output the answer directly:
1. `<answer>X</answer>`: The final answer, where X should be A, B, C, or D.
Output format:  
<answer>X</answer> The final answer, where X should be A, B, C, or D.

Note:
1. No explanation is needed.

Example:
<answer>A</answer>
"""

prompt_choice_cot = """
Please select the correct answer based on the question and the provided image, and reason step by step to output the thought process that leads to the correct answer. The output format should include two parts:
1. `<cot>reasoning process</cot>`: The step-by-step reasoning process, which may include analysis, logical reasoning, application of relevant knowledge, etc.
2. `<answer>X</answer>`: The final answer, where X should be A, B, C, or D.

Note:
Please reason step by step, output the intermediate reasoning process, and finally provide the answer.  

Example: 
<cot>The question stem graphics are divided into two groups, top and bottom. It is observed that the first group of graphics are all letters, and the rightmost letter in the previous graphic along the arrow direction is on the leftmost side of the adjacent next graphic. The second set of figures are all numbers, and the rightmost number in the previous figure in the direction of the arrow is on the leftmost side of the adjacent figure. According to this rule,? The leftmost digit should be 2 and the rightmost digit should be 4, only option A matches.</cot>  
<answer>A</answer>
"""

prompt_sudoku_direct='''
Please solve the following Sudoku puzzle. The basic requirement of Sudoku is to fill each empty space with any Arabic numeral from 1 to 9  such that in each row, each column, and each adjacent 3×3 small square of the same color, the numbers filled in must be from 1 to 9 without any repetition.

Output format:  
<answer>X</answer>The final answer, where X should be the answer of Sudoku. No explanation is needed.

Note:
1. `<answer>X</answer>`: The final answer, where X should be the answer of Sudoku.
2. the final answer must be in <answer> </answer>
3. No explanation is needed.

Example:
<answer>576891432\n243567198\n819234765\n354678219\n687912543\n921345876\n798123654\n465789321\n132456987</answer>
'''

prompt_sudoku_cot='''
Please solve the following Sudoku puzzle. The basic requirement of Sudoku is to fill each empty space with any Arabic numeral from 1 to 9  such that in each row, each column, and each adjacent 3×3 small square of the same color, the numbers filled in must be from 1 to 9 without any repetition.

Output format:  
Step-by-step thinking process

<answer>X</answer>The final answer, where X should be the answer of Sudoku.

Note:
1. `<answer>X</answer>`: The final answer, where X should be the answer of Sudoku.
2. the final answer must be in <answer> </answer>

Example:
<cot>Given the input [1, 0, 3, 0, 0, 0, 2, 0, 0], the first row is missing 2, so it becomes [1, 2, 3]. Column 1 has 1 and 2, so the next value is 3; column 2 has 2, so the next is 1; and the remaining number in row 2 is 2. In row 3, column 2 needs 3 and column 3 needs 1. The completed grid is [1, 2, 3, 3, 1, 2, 2, 3, 1].</cot>
<answer>123312231</answer>
'''

prompt_raven_direct='''Please solve the following raven puzzle. 
Output format:  
<answer>X</answer>The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.

Note:
1. `<answer>X</answer>`: The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.
2. the final answer must be in <answer> </answer>
3. No explanation is needed.

Example:
<answer>3</answer>

'''

prompt_raven_cot='''Please solve the following raven puzzle. 
Output format:  
Describe the fine-grained description model and option diagram. Analyze the patterns in the diagram to answer the question.

<answer>X</answer>The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.

Note:
1. `<answer>X</answer>`: The final answer, where X should be the position of the subgraph in the option graph, calculated as the number of subgraphs from left to right, top to bottom, and is an Arabic numeral.
2. the final answer must be in <answer> </answer>

Example:
<cot>### The model diagram and option diagram include the following basic graphics:... \n### Description of the model diagram:\nContent of the 1st:...\nContent of the n-th:...\n### Description of the option diagram:\nContent of the 1st:... \nContent of the n-th....\n\n### Answer:3\n</cot>
<answer>3</answer>

'''

