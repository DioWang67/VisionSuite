import sys
import logging
import argparse
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


class VisionSuiteUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VisionSuite 圖形介面")
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # 配置文件選擇
        config_layout = QHBoxLayout()
        self.config_label = QLabel("使用預設 config.yaml")
        config_btn = QPushButton("選擇配置檔")
        config_btn.clicked.connect(self.select_config)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(config_btn)
        layout.addLayout(config_layout)

        # 任務勾選框
        self.tasks = {
            "format_conversion": QCheckBox("格式轉換"),
            "anomaly_detection": QCheckBox("異常檢測"),
            "yolo_augmentation": QCheckBox("YOLO 增強"),
            "image_augmentation": QCheckBox("圖像增強"),
            "dataset_splitter": QCheckBox("資料集分割"),
        }
        for cb in self.tasks.values():
            layout.addWidget(cb)

        # 任務批次操作按鈕
        task_btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全選")
        select_all_btn.clicked.connect(self.select_all_tasks)
        clear_all_btn = QPushButton("全清")
        clear_all_btn.clicked.connect(self.clear_all_tasks)
        task_btn_layout.addWidget(select_all_btn)
        task_btn_layout.addWidget(clear_all_btn)
        layout.addLayout(task_btn_layout)

        # 格式覆寫輸入
        format_layout = QHBoxLayout()
        self.input_format_edit = QLineEdit()
        self.input_format_edit.setPlaceholderText(".bmp")
        self.output_format_edit = QLineEdit()
        self.output_format_edit.setPlaceholderText(".png")
        format_layout.addWidget(QLabel("輸入格式"))
        format_layout.addWidget(self.input_format_edit)
        format_layout.addWidget(QLabel("輸出格式"))
        format_layout.addWidget(self.output_format_edit)
        layout.addLayout(format_layout)

        # 執行按鈕
        run_btn = QPushButton("執行流程")
        run_btn.clicked.connect(self.execute_pipeline)
        layout.addWidget(run_btn)

        # 日誌輸出區域
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.config_path = "config.yaml"

    def select_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置檔", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            self.config_path = path
            self.config_label.setText(path)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = VisionSuiteUI()
    ui.show()
    sys.exit(app.exec_())
