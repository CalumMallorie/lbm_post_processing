import os
import xml.etree.ElementTree as ET

class ParameterReader:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def parse_xml_content(self, content):
        cleaned_content = content.replace('<?xml version="1.0" ?>', '').strip()
        wrapped_content = f"<root>{cleaned_content}</root>"
        return ET.fromstring(wrapped_content)

    def extract_parameter_value(self, root, path):
        current_element = root
        for tag in path[:-1]:
            current_element = current_element.find(tag)
            if current_element is None:
                return None
        return current_element.get(path[-1])

    def get_value_from_files(self, parameter_path):
        for root_dir, _, files in os.walk(self.directory_path):
            for file in files:
                if file.endswith(".xml"):
                    file_path = os.path.join(root_dir, file)
                    with open(file_path, 'r') as xml_file:
                        content = xml_file.read()
                        parsed_content = self.parse_xml_content(content)
                        parameter_value = self.extract_parameter_value(parsed_content, parameter_path)
                        if parameter_value:
                            return float(parameter_value)
        return None

