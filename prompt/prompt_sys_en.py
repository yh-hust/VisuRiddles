prompt_choice_direct = """
Please select the correct answer based on the question and the provided image, and output the answer.
The final answer should be provided as follows:
<answer>X</answer>: The final answer, where X should be one of the answer choices A, B, C, or D.

Note:
1. The final answer must be in <answer> </answer>.
2. `<answer>X</answer>`: The final answer, where X should be A, B, C, or D.
"""

prompt_choice_cot = '''
Please select the correct answer based on the question and the provided image, and output the answer.

<answer>X</answer>: The final answer, where X should be A, B, C, or D.

Follow these steps to solve the problem:

1. **Analyze the Image**: Look at the image and identify key details.
2. **Summarize the Pattern**: Find any patterns or relationships in the image.
3. **Evaluate the Options**: Review the answer choices and see which one matches the pattern.
4. **Choose the Best Answer**: Select the answer that best fits the pattern.

Note:
1. The final answer must be in <answer> </answer>.
2. `<answer>X</answer>`: The final answer, where X should be A, B, C, or D.
'''

prompt_sudoku_direct='''
Please solve the following Sudoku puzzle. The basic requirement of Sudoku is to fill each empty space with any Arabic numeral from 1 to 9 such that in each row, each column, and each adjacent 3×3 small square of the same color, the numbers filled in must be from 1 to 9 without any repetition.

Note:
1. The final answer must be provided in the format `<answer>X</answer>`, where X is the solved Sudoku grid.
2. The solution should be in the form of a 9x9 grid, where each row contains exactly 9 digits (from 1 to 9) with no spaces between them, and rows are separated by a newline character (`\n`).
3. The final answer must be enclosed within `<answer> </answer>`.
'''
#<answer>576891432\n243567198\n819234765\n354678219\n687912543\n921345876\n798123654\n465789321\n132456987</answer>
prompt_sudoku_cot='''
Please solve the following Sudoku puzzle. The basic requirement of Sudoku is to fill each empty space with any Arabic numeral from 1 to 9 such that in each row, each column, and each adjacent 3×3 small square of the same color, the numbers filled in must be from 1 to 9 without any repetition.

Follow these steps:

1. **Analyze the Initial Grid**: Look at the given Sudoku puzzle. Identify which cells are already filled with numbers, and which ones are empty. Also, notice the numbers in each row, column, and 3x3 subgrid.

2. **Identify the Constraints**: Each row, each column, and each 3x3 subgrid must contain the numbers 1 through 9 without repetition. Based on this rule, identify which numbers are missing from the rows, columns, or subgrids.

3. **Start with Obvious Placements**: Look for rows, columns, or subgrids where there is only one possible place for a missing number. Fill these cells in first.

4. **Use Logical Deduction**: If you cannot immediately fill in a number, use logical reasoning to eliminate possibilities. For each empty cell, consider the numbers already present in the same row, column, and 3x3 subgrid, and narrow down the potential candidates for that cell.

5. **Iterate and Fill in More Numbers**: As you fill in more numbers, you will uncover more obvious placements. Continue using logical deduction to fill in the grid.

6. **Complete the Puzzle**: Continue applying these steps until the Sudoku grid is fully solved.

Note:
1. The final answer must be provided in the format `<answer>X</answer>`, where X is the solved Sudoku grid.
2. The solution should be in the form of a 9x9 grid, where each row contains exactly 9 digits (from 1 to 9) with no spaces between them, and rows are separated by a newline character (`\n`).
3. The final answer must be enclosed within `<answer> </answer>`.
'''

prompt_raven_direct='''Please solve the following raven puzzle. 
Note: 
1. The final answer must be in <answer> </answer>.  
2. <answer>X</answer>: The final answer, where X should be exactly one of the option labels A–H.  
'''

prompt_raven_cot='''Please solve the following raven puzzle. Follow these steps:
1. Analyze Elements: Look at all the shapes and their characteristics (e.g., size, color, position, quantity).
2. Identify Patterns: Find any relationships or patterns between the shapes (e.g., changes in size, position, or color).
3. Form a Hypothesis: Based on the pattern, guess what the next shape should be.
4. Derive the Answer: Use your hypothesis to find the correct answer, making sure it matches the pattern.
Note: 
1. <answer>X</answer>: The final answer, where X should be exactly one of the option labels A–H.  
2. the final answer must be in <answer> </answer>
'''

