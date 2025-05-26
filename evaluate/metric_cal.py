import json
import os
import re 

def Choice_evalution(item):
    if not isinstance(item['pred'],str):
        return 0
    matches = re.findall(r"<answer>([A-D])</answer>", item['pred'], re.IGNORECASE)
    if not matches:
        return 0

    pred_answer = matches[-1].upper()
    gold_answer = item.get('gold_answer', '').upper()
    
    return 1 if pred_answer == gold_answer else 0

def Sudoku_evalution(item):
    if not isinstance(item['pred'],str):
        return 0
    matches = re.findall(r"<answer>(.*?)</answer>", item.get('pred', ''), re.DOTALL)
    if not matches:
        print(item.get('pred', ''))
        return 0

    pred_content = matches[-1].lstrip('\n').rstrip('\n')
    pred_content = ''.join(c for c in pred_content if c.isdigit() or c == '\n')
    gold_content = item.get('gold_answer', '')
    
    gold_lines = [line.strip() for line in gold_content.strip().split('\n') if line.strip()]
    if '\n' in pred_content:
        pred_lines = [line.strip() for line in pred_content.strip().split('\n') if line.strip()]
    else:
        pred_lines = pred_content.strip()
        pred_lines = [pred_lines[i:i+len(gold_lines[0])] for i in range(0, len(pred_lines), len(gold_lines[0]))]

    if len(pred_lines) != len(gold_lines):
        print(f'matches:{matches[-1]}')
        print(f'pred_lines:{pred_lines}')
        print(f'gold_lines:{gold_lines}')
        return 0
    
    for pred_line, gold_line in zip(pred_lines, gold_lines):
        if pred_line != gold_line:
            if len(pred_line) != len(gold_line):
                print(pred_lines)
                print(gold_lines)
            return 0

    return 1

def Raven_evalution(item):
    if not isinstance(item['pred'],str):
        return 0
    matches = re.findall(r"<answer>(\d+)</answer>", item.get('pred', ''), re.IGNORECASE)
    if not matches:
        return 0
    pred_answer = matches[-1].strip()
    gold_answer = str(item.get('gold_answer', '')).strip()
    return 1 if pred_answer == gold_answer else 0

if __name__=='__main__':
    json_path = r'/path/to/your/result/file.json'
    data_list = json.load(open(json_path))
    result_dict = {
        'Numerical':[],
        'Stylistic':[],
        'Attribute':[],
        'Positional':[],
        'Spatial':[],
        'sudoku':[],
        'raven':[],
        'Other':[],
        'All':[]
    }
    for item in data_list:
        acc = 0
        if item['class'] == 'raven':
            acc = Raven_evalution(item)
        elif item['class'] == 'sudoku':
            acc = Sudoku_evalution(item)
        else:
            acc = Choice_evalution(item)
        result_dict[item['class']].append(acc)
        result_dict['All'].append(acc)
    acc_list = []
    for category, values in result_dict.items():
        if values:
            avg = sum(values) / len(values)*100
        else:
            avg = 0.0
        print(f"{category}: mean = {avg:.2f}")
        acc_list.append(avg)
    acc_list = [f"{v:.2f}" for v in acc_list]
    print('&'.join(acc_list))
    
        
