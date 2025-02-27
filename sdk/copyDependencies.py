import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# 用于存储已处理的文件，避免重复处理
processed_files = set()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="分析文件依赖并复制到目标目录，同时更新导入路径"
    )
    parser.add_argument("--input", required=True, help="要分析的文件路径")
    parser.add_argument("--output", default="src/main", help="目标目录路径")
    parser.add_argument("--prefix", help="导入语句的新前缀")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    # 将输出路径转换为绝对路径
    output_dir = os.path.abspath(args.output)

    # 如果没有指定前缀，使用输出目录的基本名称作为前缀
    prefix = args.prefix or os.path.basename(output_dir)

    return args.input, output_dir, prefix, args.verbose


def find_module_file(module_name, search_paths, verbose=False):
    """查找模块的文件路径"""
    if verbose:
        print(f"尝试查找模块: {module_name}")

    module_parts = module_name.split(".")

    # 在所有搜索路径中查找
    for search_path in search_paths:
        # 构建可能的路径
        module_path = os.path.join(search_path, *module_parts)

        # 尝试.py文件
        for ext in [".py", ".pyx", ".pyd", ".so", ".pyw"]:
            file_path = module_path + ext
            if os.path.isfile(file_path):
                if verbose:
                    print(f"找到模块: {file_path}")
                return file_path

        # 尝试作为包（__init__.py）
        init_path = os.path.join(module_path, "__init__.py")
        if os.path.isfile(init_path):
            if verbose:
                print(f"找到包: {init_path}")
            return init_path

    if verbose:
        print(f"未找到模块: {module_name}")
    return None


def parse_imports(file_path, verbose=False):
    """解析文件中的导入语句"""
    imports = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"无法读取文件 {file_path}: {str(e)}")
        return imports, ""

    # 解析 import 语句
    import_regex = re.compile(
        r"^import\s+([\w\.,\s]+)(?:\s+as\s+[\w]+)?$", re.MULTILINE
    )
    for match in import_regex.finditer(content):
        for module in match.group(1).split(","):
            module = module.strip()
            if module:
                imports.append(module)
                if verbose:
                    print(f"找到导入: import {module}")

    # 解析 from ... import ... 语句
    from_regex = re.compile(r"^from\s+([\w\.]+)\s+import\s+([\w\s,*]+)$", re.MULTILINE)
    for match in from_regex.finditer(content):
        module = match.group(1).strip()
        if module:
            imports.append(module)
            if verbose:
                print(f"找到导入: from {module} import ...")

    return imports, content


def copy_module_with_structure(src_file, module_name, target_dir, verbose=False):
    """复制模块并保持目录结构"""
    # 创建目标目录结构
    module_parts = module_name.split(".")
    module_dir = (
        os.path.join(target_dir, *module_parts[:-1])
        if len(module_parts) > 1
        else target_dir
    )
    os.makedirs(module_dir, exist_ok=True)

    # 目标文件路径
    if os.path.basename(src_file) == "__init__.py":
        target_file = os.path.join(module_dir, "__init__.py")
    else:
        ext = os.path.splitext(src_file)[1]
        target_file = os.path.join(module_dir, f"{module_parts[-1]}{ext}")

    # 复制文件
    if not os.path.exists(target_file):
        try:
            shutil.copy2(src_file, target_file)
            if verbose:
                print(f"复制文件: {src_file} -> {target_file}")
            return target_file
        except Exception as e:
            print(f"复制文件失败 {src_file} -> {target_file}: {str(e)}")
    elif verbose:
        print(f"文件已存在: {target_file}")

    return target_file


def rewrite_imports(file_path, prefix, verbose=False):
    """重写文件中的导入语句，添加前缀"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"无法读取文件进行重写 {file_path}: {str(e)}")
        return

    # 重写 import 语句
    def rewrite_import(match):
        modules = match.group(1)
        # 处理多个导入，如 import x, y, z
        new_modules = []
        for module in modules.split(","):
            module = module.strip()
            if module and not module.startswith("."):  # 不处理相对导入
                new_modules.append(f"{prefix}.{module}")
            else:
                new_modules.append(module)
        return f"import {', '.join(new_modules)}"

    modified_content = re.sub(
        r"^import\s+([\w\.,\s]+)(?:\s+as\s+[\w]+)?$",
        rewrite_import,
        content,
        flags=re.MULTILINE,
    )

    # 重写 from ... import ... 语句
    def rewrite_from_import(match):
        module = match.group(1).strip()
        imports = match.group(2)
        if module and not module.startswith("."):  # 不处理相对导入
            return f"from {prefix}.{module} import {imports}"
        return match.group(0)

    modified_content = re.sub(
        r"^from\s+([\w\.]+)\s+import\s+([\w\s,*]+)$",
        rewrite_from_import,
        modified_content,
        flags=re.MULTILINE,
    )

    # 写回文件
    if content != modified_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            if verbose:
                print(f"重写导入: {file_path}")
        except Exception as e:
            print(f"重写文件失败 {file_path}: {str(e)}")


def analyze_dependencies(
    file_path,
    module_name,
    target_dir,
    prefix,
    search_paths,
    depth=0,
    max_depth=5,
    verbose=False,
):
    """递归分析并复制依赖"""
    if file_path in processed_files or depth > max_depth:
        return

    processed_files.add(file_path)

    if verbose:
        print(f"\n分析 [{depth}]: {module_name} -> {file_path}")

    # 复制当前模块
    target_file = copy_module_with_structure(
        file_path, module_name, target_dir, verbose
    )

    # 解析导入
    imports, _ = parse_imports(file_path, verbose)

    # 处理每个导入
    for imported_module in imports:
        # 跳过标准库
        if imported_module in sys.builtin_module_names:
            continue

        # 处理相对导入
        if imported_module.startswith("."):
            if verbose:
                print(f"跳过相对导入: {imported_module}")
            continue

        # 查找模块文件
        imported_file = find_module_file(imported_module, search_paths, verbose)
        if imported_file:
            analyze_dependencies(
                imported_file,
                imported_module,
                target_dir,
                prefix,
                search_paths,
                depth + 1,
                max_depth,
                verbose,
            )


def main():
    # 解析命令行参数
    input_file, output_dir, prefix, verbose = parse_arguments()

    # 项目根目录
    project_dir = os.getcwd()

    # 获取Python搜索路径
    search_paths = [project_dir] + sys.path

    # 解析输入文件
    abs_input_file = os.path.join(project_dir, input_file)
    input_module_name = os.path.splitext(input_file.replace("/", "."))[0]

    if not os.path.exists(abs_input_file):
        print(f"错误: 文件不存在 {abs_input_file}")
        sys.exit(1)

    print(f"项目目录: {project_dir}")
    print(f"输出目录: {output_dir}")
    print(f"导入前缀: {prefix}")
    print(f"输入文件: {abs_input_file}")
    print(f"输入模块: {input_module_name}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 分析依赖并复制
    analyze_dependencies(
        abs_input_file,
        input_module_name,
        output_dir,
        prefix,
        search_paths,
        verbose=verbose,
    )

    print(f"依赖复制完成！共处理了 {len(processed_files)} 个文件")
    print(f"所有导入语句已更新为使用前缀: {prefix}")


if __name__ == "__main__":
    main()
