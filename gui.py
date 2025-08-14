import sys
import logging
import argparse
import yaml
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QCheckBox,
    QTextEdit,
    QMessageBox,
    QHBoxLayout,
    QLineEdit,
    QGroupBox,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
)

from main_pipeline import load_config, setup_logging, run_pipeline


class QTextEditLogger(logging.Handler):
    """將日誌輸出導向 QTextEdit""" 

    def __init__(self, widget: QTextEdit):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.append(msg)


class ConfigEditorDialog(QDialog):
    """以表單方式編輯 config.yaml"""

    def __init__(self, config_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.setWindowTitle("編輯配置")
        self.resize(500, 600)
        self.setStyleSheet(
            """
            QWidget{font-size:14px;}
            QGroupBox{font-weight:bold; border:1px solid #cccccc; margin-top:10px;}
            QGroupBox::title{subcontrol-origin: margin; left:10px; padding:0 3px 0 3px;}
            QLineEdit{padding:4px;}
            """
        )

        self.config = load_config(config_path)

        layout = QVBoxLayout(self)

        # 流水線設定
        pipeline_group = QGroupBox("流水線")
        pipeline_layout = QFormLayout()
        self.log_file_edit = QLineEdit(self.config["pipeline"].get("log_file", ""))
        pipeline_layout.addRow("日誌檔案", self.log_file_edit)
        pipeline_group.setLayout(pipeline_layout)
        layout.addWidget(pipeline_group)

        # 任務啟用
        tasks_group = QGroupBox("任務啟用")
        tasks_layout = QVBoxLayout()
        self.task_checkboxes: dict[str, QCheckBox] = {}
        for task in self.config["pipeline"]["tasks"]:
            cb = QCheckBox(task["name"])
            cb.setChecked(task.get("enabled", True))
            self.task_checkboxes[task["name"]] = cb
            tasks_layout.addWidget(cb)
        tasks_group.setLayout(tasks_layout)
        layout.addWidget(tasks_group)

        # 格式轉換設定
        fc = self.config.get("format_conversion", {})
        fc_group = QGroupBox("格式轉換")
        fc_layout = QFormLayout()
        self.fc_input_dir = QLineEdit(fc.get("input_dir", ""))
        self.fc_output_dir = QLineEdit(fc.get("output_dir", ""))
        self.fc_input_formats = QLineEdit(", ".join(fc.get("input_formats", [])))
        self.fc_output_format = QLineEdit(fc.get("output_format", ""))
        fc_layout.addRow("輸入資料夾", self.fc_input_dir)
        fc_layout.addRow("輸出資料夾", self.fc_output_dir)
        fc_layout.addRow("輸入格式", self.fc_input_formats)
        fc_layout.addRow("輸出格式", self.fc_output_format)
        fc_group.setLayout(fc_layout)
        layout.addWidget(fc_group)

        # 按鈕
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.save)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def save(self) -> None:
        self.config["pipeline"]["log_file"] = self.log_file_edit.text()
        for task in self.config["pipeline"]["tasks"]:
            task["enabled"] = self.task_checkboxes[task["name"]].isChecked()

        fc = self.config.setdefault("format_conversion", {})
        fc["input_dir"] = self.fc_input_dir.text()
        fc["output_dir"] = self.fc_output_dir.text()
        fc["input_formats"] = [
            s.strip() for s in self.fc_input_formats.text().split(",") if s.strip()
        ]
        fc["output_format"] = self.fc_output_format.text()

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)
        self.accept()

class VisionSuiteUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VisionSuite 圖形介面")
        self.resize(500, 400)
        self.setStyleSheet(
            """
            QWidget{font-size:14px;}
            QGroupBox{font-weight:bold; border:1px solid #cccccc; margin-top:10px;}
            QGroupBox::title{subcontrol-origin: margin; left:10px; padding:0 3px 0 3px;}
            QPushButton{padding:6px 12px;}
            """
        )

        layout = QVBoxLayout(self)

        # 配置文件選擇
        config_layout = QHBoxLayout()
        self.config_label = QLabel("使用預設 config.yaml")
        config_btn = QPushButton("選擇配置檔")
        config_btn.clicked.connect(self.select_config)
        edit_btn = QPushButton("編輯配置")
        edit_btn.clicked.connect(self.edit_config)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(config_btn)
        config_layout.addWidget(edit_btn)
        layout.addLayout(config_layout)

        # 任務勾選框
        task_group = QGroupBox("任務選擇")
        task_layout = QVBoxLayout()
        self.tasks = {
            "format_conversion": QCheckBox("格式轉換"),
            "anomaly_detection": QCheckBox("異常檢測"),
            "yolo_augmentation": QCheckBox("YOLO 增強"),
            "image_augmentation": QCheckBox("圖像增強"),
            "dataset_splitter": QCheckBox("資料集分割"),
        }
        for cb in self.tasks.values():
            task_layout.addWidget(cb)

        # 任務批次操作按鈕
        task_btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全選")
        select_all_btn.clicked.connect(self.select_all_tasks)
        clear_all_btn = QPushButton("全清")
        clear_all_btn.clicked.connect(self.clear_all_tasks)
        task_btn_layout.addWidget(select_all_btn)
        task_btn_layout.addWidget(clear_all_btn)
        task_layout.addLayout(task_btn_layout)
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        # 格式覆寫輸入
        format_group = QGroupBox("格式覆寫")
        format_layout = QHBoxLayout()
        self.input_format_edit = QLineEdit()
        self.input_format_edit.setPlaceholderText(".bmp")
        self.output_format_edit = QLineEdit()
        self.output_format_edit.setPlaceholderText(".png")
        format_layout.addWidget(QLabel("輸入格式"))
        format_layout.addWidget(self.input_format_edit)
        format_layout.addWidget(QLabel("輸出格式"))
        format_layout.addWidget(self.output_format_edit)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # 執行與日誌操作按鈕
        run_layout = QHBoxLayout()
        run_btn = QPushButton("執行流程")
        run_btn.clicked.connect(self.execute_pipeline)
        clear_log_btn = QPushButton("清除日誌")
        clear_log_btn.clicked.connect(self.log_clear)
        run_layout.addWidget(run_btn)
        run_layout.addWidget(clear_log_btn)
        layout.addLayout(run_layout)

        # 日誌輸出區域
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "font-family: Consolas, monospace; background:#f9f9f9;"
        )
        layout.addWidget(self.log_output)

        self.config_path = "config.yaml"
        self.apply_config_defaults()

    def select_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置檔", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            self.config_path = path
            self.config_label.setText(path)
            self.apply_config_defaults()

    def apply_config_defaults(self) -> None:
        try:
            config = load_config(self.config_path)
            task_configs = {
                t["name"]: t.get("enabled", True) for t in config["pipeline"]["tasks"]
            }
            for name, cb in self.tasks.items():
                cb.setChecked(task_configs.get(name, False))

            fc = config.get("format_conversion", {})
            if fc.get("input_formats"):
                self.input_format_edit.setPlaceholderText(fc["input_formats"][0])
            if fc.get("output_format"):
                self.output_format_edit.setPlaceholderText(fc["output_format"])
        except Exception as e:
            QMessageBox.warning(self, "警告", f"無法讀取配置: {e}")

    def edit_config(self) -> None:
        dialog = ConfigEditorDialog(self.config_path, self)
        if dialog.exec_():
            self.apply_config_defaults()

    def execute_pipeline(self) -> None:
        selected_tasks = [name for name, cb in self.tasks.items() if cb.isChecked()]
        if not selected_tasks:
            QMessageBox.warning(self, "提示", "請至少選擇一個任務")
            return

        self.log_output.clear()
        try:
            config = load_config(self.config_path)
            logger = setup_logging(config["pipeline"]["log_file"])
            text_handler = QTextEditLogger(self.log_output)
            text_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(text_handler)

            args = argparse.Namespace(
                config=self.config_path,
                input_format=self.input_format_edit.text().strip() or None,
                output_format=self.output_format_edit.text().strip() or None,
            )

            run_pipeline(selected_tasks, config, logger, args)
            QMessageBox.information(self, "完成", "流程執行完成")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))
        finally:
            logger.removeHandler(text_handler)

    def select_all_tasks(self) -> None:
        for cb in self.tasks.values():
            cb.setChecked(True)

    def clear_all_tasks(self) -> None:
        for cb in self.tasks.values():
            cb.setChecked(False)

    def log_clear(self) -> None:
        self.log_output.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = VisionSuiteUI()
    ui.show()
    sys.exit(app.exec_())

