import os
import xml.etree.ElementTree as ET

class DirectoryScanner:
    def __init__(self, filepath):
        self.filepath = filepath

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

    def scan_directory_for_specific_parameter(self, directory_path, parameter_path):
        results = {}
        for root_dir, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".xml"):
                    file_path = os.path.join(root_dir, file)
                    with open(file_path, 'r') as xml_file:
                        content = xml_file.read()
                        parsed_content = self.parse_xml_content(content)
                        parameter_value = self.extract_parameter_value(parsed_content, parameter_path)
                        if parameter_value:
                            # Storing the directory path instead of the full XML file path
                            results[root_dir] = parameter_value
        return results

    def query_files_by_parameter(self, xml_data, parameter_value):
        return [path for path, value in xml_data.items() if value == parameter_value]

    def get_directories(self, directory_path, parameter_path, desired_value):
        xml_data = self.scan_directory_for_specific_parameter(directory_path, parameter_path)
        queried_paths = self.query_files_by_parameter(xml_data, desired_value)
        return queried_paths
    
    def get_directories_containing_file(self, target_file='parameters.xml'):
        """
        Traverses the directory tree rooted at self.filepath and returns a list of directories
        containing the specified target_file.

        Parameters:
            target_file (str): The name of the file to search for in each directory.

        Returns:
            list: A list of directories containing target_file.
        """
        directories_containing_target = []
        for root, dirs, files in os.walk(self.filepath):
            if target_file in files:
                directories_containing_target.append(root)
        return directories_containing_target