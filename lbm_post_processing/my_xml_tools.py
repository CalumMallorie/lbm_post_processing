"""
my_xml_tools.py
===============

This module provides tools for reading and parsing XML files within a
specified directory. It includes the `ReadSimXML` class designed to
navigate a directory, read XML files, and extract specific parameter
values based on a provided path.

Classes
-------
ReadSimXML
    Reads and parses XML files within a specified directory and extracts
    parameter values based on a provided path.

Dependencies
------------
- os
- xml.etree.ElementTree as ET

Functions and Methods
---------------------
ReadSimXML
    __init__(self, directory_path)
        Initializes the ReadSimXML class with a directory path.
    parse_xml_content(self, content)
        Parses XML content from a string.
    extract_parameter_value(self, root, path)
        Extracts a parameter value from the XML content.
    search(self, parameter_path)
        Searches for a parameter value in all XML files within the
        directory.

Example Usage
-------------
    from my_xml_tools import ReadSimXML
    
    # Initialize with path to XML directory
    xml_reader = ReadSimXML('path/to/xml/directory')
    
    # Search for a specific parameter value
    parameter_value = xml_reader.search(['LBM', 'relaxation', 'tau'])
"""

import os
import xml.etree.ElementTree as ET

class ReadSimXML:
    """Read and parse XML files within a specified directory.

    This class is designed to navigate a directory, read XML files,
    and extract specific parameter values based on a provided path.

    Attributes
    ----------
    directory_path : str
        Path to the directory containing XML files.
    """

    def __init__(self, directory_path: str):
        """Initialize the ReadSimXML class with a directory path.

        Args
        ----
        directory_path : str
            The path to the directory containing XML files.

        Raises
        ------
        ValueError
            If the provided directory path does not exist.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory path '{directory_path}' does not exist.")
        self.directory_path = directory_path

    def parse_xml_content(self, content: str) -> ET.ElementTree:
        """Parse XML content from a string. The content must be wrapped
        in a root tag first, as the BioFM XML files do not have one.

        Args
        ----
        content : str
            XML content in string format.

        Returns
        -------
        ET.ElementTree
            Parsed XML content as an ElementTree object.

        Raises
        ------
        ET.ParseError
            If the content is not valid XML.
        """
        try:
            cleaned_content = content.replace('<?xml version="1.0" ?>', '').strip()
            wrapped_content = f"<root>{cleaned_content}</root>"
            return ET.fromstring(wrapped_content)
        except ET.ParseError as e:
            raise ET.ParseError(f"Error parsing XML content: {e}")

    def extract_parameter_value(self, root: ET.Element, path: list) -> str:
        """Extract a parameter value from the XML content.

        Traverses the XML tree according to the specified path and retrieves
        the value of the desired parameter.

        Args
        ----
        root : ET.Element
            The root element of the XML tree.
        path : list
            The path to the parameter within the XML tree.

        Returns
        -------
        str
            The value of the parameter, or None if not found.
        """
        current_element = root
        for tag in path[:-1]:
            current_element = current_element.find(tag)
            if current_element is None:
                return None
        return current_element.get(path[-1])

    def search(self, parameter_path: list) -> float:
        """Search for a parameter value in all XML files within the directory.

        Iterates over all XML files in the directory and searches for the
        specified parameter.

        Args
        ----
        parameter_path : list
            The path to the parameter in the XML structure.

        Returns
        -------
        float
            The value of the parameter as a float, or None if not found.

        Raises
        ------
        FileNotFoundError
            If no XML files are found in the directory.
        ValueError
            If the parameter value cannot be converted to a float.
        """
        xml_files_found = False
        for root_dir, _, files in os.walk(self.directory_path):
            for file in files:
                if file.endswith(".xml") and not file.startswith('.'):
                    xml_files_found = True
                    file_path = os.path.join(root_dir, file)
                    try:
                        with open(file_path, 'r') as xml_file:
                            content = xml_file.read()
                            parsed_content = self.parse_xml_content(content)
                            parameter_value = self.extract_parameter_value(parsed_content, parameter_path)
                            if parameter_value is not None:
                                return float(parameter_value)
                    except IOError as e:
                        raise IOError(f"Error reading file {file_path}: {e}")
                    except ValueError as e:
                        raise ValueError(f"Error converting parameter value to float: {e}")

        if not xml_files_found:
            raise FileNotFoundError("No XML files found in the specified directory.")

        return None
