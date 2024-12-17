from kag.builder.component.reader.markdown_reader import MarkDownReader



if __name__ == "__main__":
    
    md_reader=MarkDownReader(cut_depth=1,project_id=1).invoke(input='/home/luocheng/project/KAG/tests/builder/data/test_markdown.md')