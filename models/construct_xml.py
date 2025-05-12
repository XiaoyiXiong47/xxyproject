"""
Created by: Xiaoyi Xiong
Date: 13/05/2025
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def save_pretty_xml(xml_tree, filename="hand_angles.xml"):
    xml_str = ET.tostring(xml_tree.getroot(), encoding="utf-8")

    dom = minidom.parseString(xml_str)
    pretty_xml_as_string = dom.toprettyxml(indent="  ")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_string)

    print(f"{filename} saved!")


def write_xml(file, element):
    """Initialise the XML file"""
    # create root element
    root = ET.Element("doc")

    # create sem
    sem = ET.SubElement(root, "sem")
    sem.attrib["type"] = element["type"]
    # ...

    # Create the tree and write to a file
    tree = ET.ElementTree(root)
    tree.write(file)


def append_element_to_xml(file, element):
    """ Append element into XML file"""
    # Load the existing XML file
    tree = ET.parse(file)
    root = tree.getroot()

    # Append new data
    new_sem = ET.SubElement(root, "sem")
    new_sem.attrib["type"] = element["type"]
    # ...

    # Save the changes back to the file
    tree.write(file)



