import xml.etree.ElementTree as ET


def read_xml_file(file_path):
    try:
        # ?? XML ??
        tree = ET.parse(file_path)
        root = tree.getroot()

        # ?? XML ????
        print("XML ???")
        for element in root:
            print(element.tag, element.text)

        # ????????????????
        print("\n?????")
        for child in root:
            print(f"Tag: {child.tag}, Attributes: {child.attrib}")
            for subchild in child:
                print(f"  SubTag: {subchild.tag}, Text: {subchild.text}")

    except FileNotFoundError:
        print(f"12 {file_path} ????")
    except ET.ParseError:
        print(f"32 {file_path} ?????")


# ????
file_path = '../datasets/TCAL/ANNOTATIONS/lable1_1.xml'
read_xml_file(file_path)
