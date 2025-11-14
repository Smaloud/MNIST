import json
import argparse
import os
import sys

def notebook_to_python(ipynb_path, py_path=None, include_markdown=True):
    """
    将Jupyter Notebook文件转换为Python文件
    
    参数:
        ipynb_path (str): 输入的.ipynb文件路径
        py_path (str, 可选): 输出的.py文件路径，默认与.ipynb同目录同名称
        include_markdown (bool): 是否将Markdown单元格转为注释，默认True
    """
    # 验证输入文件
    if not os.path.exists(ipynb_path):
        raise FileNotFoundError(f"文件不存在: {ipynb_path}")
    
    if not ipynb_path.endswith(".ipynb"):
        raise ValueError("输入文件必须是.ipynb格式")
    
    # 处理输出路径
    if py_path is None:
        # 默认输出路径：与输入文件同目录，替换后缀为.py
        py_path = os.path.splitext(ipynb_path)[0] + ".py"
    
    # 读取并解析Notebook文件
    try:
        with open(ipynb_path, "r", encoding="utf-8") as f:
            notebook_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("无法解析Notebook文件，可能是损坏的JSON格式")
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {str(e)}")
    
    # 提取单元格内容
    code_lines = []
    cells = notebook_data.get("cells", [])
    
    for cell in cells:
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])  # 单元格内容（列表形式，每行一个元素）
        
        if not source:  # 跳过空单元格
            continue
        
        # 处理代码单元格
        if cell_type == "code":
            # 拼接代码行（保留原始格式）
            code_lines.extend(source)
            code_lines.append("\n")  # 单元格之间加空行分隔
        
        # 处理Markdown单元格（转为注释）
        elif cell_type == "markdown" and include_markdown:
            # 用#注释Markdown内容，前后加空行区分
            code_lines.append("\n")
            for line in source:
                # 跳过空行的注释（避免过多空注释）
                if line.strip() == "":
                    code_lines.append("\n")
                else:
                    code_lines.append(f"# {line}")
            code_lines.append("\n")
    
    # 写入Python文件
    try:
        with open(py_path, "w", encoding="utf-8") as f:
            f.writelines(code_lines)
        print(f"转换成功！已保存至: {os.path.abspath(py_path)}")
    except Exception as e:
        raise RuntimeError(f"写入文件失败: {str(e)}")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="将Jupyter Notebook转换为Python文件")
    parser.add_argument("input", help="输入的.ipynb文件路径")
    parser.add_argument("-o", "--output", help="输出的.py文件路径（可选）")
    parser.add_argument("-nm", "--no-markdown", action="store_true", 
                      help="不包含Markdown单元格内容")
    
    args = parser.parse_args()
    
    # 执行转换
    try:
        notebook_to_python(
            ipynb_path=args.input,
            py_path=args.output,
            include_markdown=not args.no_markdown
        )
    except Exception as e:
        print(f"转换失败: {str(e)}", file=sys.stderr)
        exit(1)