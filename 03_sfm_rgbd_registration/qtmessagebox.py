from PyQt5.QtWidgets import QMessageBox

def confirm_message(title, text, yes_text="Yes", no_text="No"):
    # 创建一个消息框
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle(title)
    msg.setText(text)
    # 添加按钮
    yes_button = msg.addButton(yes_text, QMessageBox.AcceptRole)
    no_button = msg.addButton(no_text, QMessageBox.RejectRole)
    # 显示消息框
    msg.exec_()
    # 判断用户点击了哪个按钮并返回
    if msg.clickedButton() == yes_button:
        return True
    elif msg.clickedButton() == no_button:
        return False
    else:
        return None


def iter_num_message(title, text):
    # 创建一个消息框
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle(title)
    msg.setText(text)
    # 添加按钮
    l20_button = msg.addButton('-20', QMessageBox.AcceptRole)
    l10_button = msg.addButton('-10', QMessageBox.AcceptRole)
    l5_button = msg.addButton('-5', QMessageBox.AcceptRole)
    l1_button = msg.addButton('-1', QMessageBox.AcceptRole)
    m_button = msg.addButton('OK', QMessageBox.RejectRole)
    r1_button = msg.addButton('+1', QMessageBox.RejectRole)
    r5_button = msg.addButton('+5', QMessageBox.RejectRole)
    r10_button = msg.addButton('+10', QMessageBox.RejectRole)
    r20_button = msg.addButton('+20', QMessageBox.RejectRole)
    # 显示消息框
    msg.exec_()
    # 判断用户点击了哪个按钮并返回
    if msg.clickedButton() == l20_button:
        return -20
    elif msg.clickedButton() == l10_button:
        return -10
    elif msg.clickedButton() == l5_button:
        return -5
    elif msg.clickedButton() == l1_button:
        return -1
    elif msg.clickedButton() == m_button:
        return 0
    elif msg.clickedButton() == r1_button:
        return 1
    elif msg.clickedButton() == r5_button:
        return 5
    elif msg.clickedButton() == r10_button:
        return 10
    elif msg.clickedButton() == r20_button:
        return 20
    else:
        return None
