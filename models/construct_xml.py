"""
Created by: Xiaoyi Xiong
Date: 13/05/2025
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np

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



def build_element(tag, content):
    # 创建元素和属性
    elem = ET.Element(tag)
    for key, value in content.items():
        if key.startswith("@"):
            attr_name = key[1:]
            elem.set(attr_name, value)
        elif isinstance(value, dict):
            child = build_element(key, value)
            elem.append(child)
        elif isinstance(value, list):
            for item in value:
                child = build_element(key, item)
                elem.append(child)
        else:
            elem.text = str(value)
    return elem

def build_xml_from_dict(root_tag, data_dict):
    root = ET.Element(root_tag)
    for tag, content in data_dict.items():
        root.append(build_element(tag, content))
    return ET.ElementTree(root)


def build_xml_frames_with_frame_index(all_angles):
    root = ET.Element("sequence")

    for idx, angle_row in enumerate(all_angles):
        frame = ET.SubElement(root, "frame", index=str(idx))  # keep frame number

        hand = ET.SubElement(frame, "hand", side="A")  # one hand

        # j1 to j3 for f0 to f4
        for i in range(5):
            j1 = angle_row[i * 3] if i * 3 < len(angle_row) else ""
            j2 = angle_row[i * 3 + 1] if i * 3 + 1 < len(angle_row) else ""
            j3 = angle_row[i * 3 + 2] if i * 3 + 2 < len(angle_row) else ""
            finger = ET.SubElement(hand, f"f{i}")
            finger.set("j1", f"{j1:.1f}" if j1 != "" and not np.isnan(j1) else "")
            finger.set("j2", f"{j2:.1f}" if j2 != "" and not np.isnan(j2) else "")
            finger.set("j3", f"{j3:.1f}" if j3 != "" and not np.isnan(j3) else "")

        # keep structure
        # ET.SubElement(hand, "orientation", xAngle="", yAngle="", zAngle="")
        # 添加 orientation
        orientation = ET.SubElement(hand, "orientation")
        orientation.set("xAngle", f"{yaw:.1f}" if not np.isnan(yaw) else "")
        orientation.set("yAngle", f"{pitch:.1f}" if not np.isnan(pitch) else "")
        orientation.set("zAngle", f"{roll:.1f}" if not np.isnan(roll) else "")
        location = ET.SubElement(hand, "location")
        ET.SubElement(location, "loc", x="", y="", z="")
        ET.SubElement(hand, "movement")

    return ET.ElementTree(root)

def xml_sign_block(dataset, gloss,hand_loc, all_angles, yaw=0, pitch=0, roll=0, side='AB', movement=None):
    """
    Build <hand> block in XML file
    :param all_angles:
    :param yaw:
    :param pitch:
    :param roll:
    :param loc:
    :return:
    """
    root = ET.Element("sem")
    root.attrib['type'] = 'sign'

    for gl in gloss:
        gloss = ET.SubElement(root, "gloss")
        gloss.attrib['lang'] = dataset
        gloss.attrib['gloss'] = gl

    for idx, angle_row in enumerate(all_angles):
        frame = ET.SubElement(root, "frame", index=str(idx))  # keep frame number

        hand = ET.SubElement(frame, "hand", side="A")  # one hand

        # j1 to j3 for f0 to f4
        for i in range(5):
            j1 = angle_row[i * 3] if i * 3 < len(angle_row) else ""
            j2 = angle_row[i * 3 + 1] if i * 3 + 1 < len(angle_row) else ""
            j3 = angle_row[i * 3 + 2] if i * 3 + 2 < len(angle_row) else ""
            finger = ET.SubElement(hand, f"f{i}")
            finger.set("j1", f"{j1:.1f}" if j1 != "" and not np.isnan(j1) else "")
            finger.set("j2", f"{j2:.1f}" if j2 != "" and not np.isnan(j2) else "")
            finger.set("j3", f"{j3:.1f}" if j3 != "" and not np.isnan(j3) else "")

    # keep structure
    # ET.SubElement(hand, "orientation", xAngle="", yAngle="", zAngle="")
    # 添加 orientation
    orientation = ET.SubElement(hand, "orientation")
    orientation.set("xAngle", f"{yaw:.1f}" if not np.isnan(yaw) else "")
    orientation.set("yAngle", f"{pitch:.1f}" if not np.isnan(pitch) else "")
    orientation.set("zAngle", f"{roll:.1f}" if not np.isnan(roll) else "")
    location = ET.SubElement(hand, "location")
    ET.SubElement(location, "loc", x=hand_loc['x'], y=hand_loc['y'], z=hand_loc['z'])
    if movement:
        ET.SubElement(hand, "movement")

    return ET.ElementTree(root)