from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from relative_distance import describe_bbox_distance


SUMMARY_HEADERS = [
    "Frame 編號",
    "蝙蝠數量",
    "中心座標 (X, Y)",
    "危險蝙蝠數",
    "確認蝙蝠總數",
    "危險進入率 (%)",
]
DETAIL_HEADERS = [
    "Frame 編號",
    "該幀蝙蝠數量",
    "偵測序號",
    "追蹤 ID",
    "中心 X",
    "中心 Y",
    "框左上 X",
    "框左上 Y",
    "框右下 X",
    "框右下 Y",
    "信心分數",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "relative_distance_score",
    "distance_bin",
    "inside_danger_zone",
    "ever_dangerous",
    "danger_bat_count",
    "confirmed_bat_total",
    "danger_entry_rate_percent",
]


class BatDetectionReport:
    def __init__(self):
        self.summary_rows = []
        self.detail_rows = []

    def add_frame(
        self,
        frame_number,
        detections,
        current_bboxes,
        danger_track_ids=None,
        inside_danger_track_ids=None,
        confirmed_total=0,
    ):
        detections = np.asarray(detections, dtype=np.float32)
        count = len(detections)
        coordinates = []
        danger_track_ids = set(danger_track_ids or [])
        inside_danger_track_ids = set(inside_danger_track_ids or [])
        confirmed_total = max(0, int(confirmed_total or 0))
        danger_count = min(len(danger_track_ids), confirmed_total)
        danger_rate = round((danger_count / confirmed_total) * 100, 2) if confirmed_total else 0.0

        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2, confidence = detection
            center_x = round(float((x1 + x2) / 2), 2)
            center_y = round(float((y1 + y2) / 2), 2)
            coordinates.append(f"({center_x:.2f}, {center_y:.2f})")
            track_id = self._match_track_id(detection[:4], current_bboxes)
            distance = describe_bbox_distance(detection[:4])
            inside_danger = track_id in inside_danger_track_ids
            ever_dangerous = track_id in danger_track_ids

            self.detail_rows.append(
                [
                    frame_number,
                    count,
                    index,
                    track_id if track_id is not None else "",
                    center_x,
                    center_y,
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                    round(float(confidence), 4),
                    distance.bbox_width,
                    distance.bbox_height,
                    distance.bbox_area,
                    distance.relative_distance_score,
                    distance.distance_bin,
                    "yes" if inside_danger else "no",
                    "yes" if ever_dangerous else "no",
                    danger_count,
                    confirmed_total,
                    danger_rate,
                ]
            )

        self.summary_rows.append(
            [
                frame_number,
                count,
                "; ".join(coordinates),
                danger_count,
                confirmed_total,
                danger_rate,
            ]
        )

    @staticmethod
    def _match_track_id(detection_bbox, current_bboxes):
        detection_center = _bbox_center(detection_bbox)
        best_id = None
        best_distance = 4.0

        for track_id, bbox in current_bboxes.items():
            distance = float(np.linalg.norm(detection_center - _bbox_center(bbox)))
            if distance <= best_distance:
                best_id = track_id
                best_distance = distance
        return best_id

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary_data = [SUMMARY_HEADERS, *self.summary_rows]
        detail_data = [DETAIL_HEADERS, *self.detail_rows]

        with ZipFile(output_path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", _content_types_xml())
            archive.writestr("_rels/.rels", _root_relationships_xml())
            archive.writestr("docProps/core.xml", _core_properties_xml())
            archive.writestr("docProps/app.xml", _app_properties_xml())
            archive.writestr("xl/workbook.xml", _workbook_xml())
            archive.writestr("xl/_rels/workbook.xml.rels", _workbook_relationships_xml())
            archive.writestr("xl/styles.xml", _styles_xml())
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                _worksheet_xml(summary_data, [14, 14, 70, 14, 16, 18]),
            )
            archive.writestr(
                "xl/worksheets/sheet2.xml",
                _worksheet_xml(
                    detail_data,
                    [
                        12,
                        12,
                        12,
                        12,
                        12,
                        12,
                        12,
                        12,
                        12,
                        12,
                        14,
                        12,
                        12,
                        12,
                        22,
                        14,
                        18,
                        16,
                        16,
                        18,
                        22,
                    ],
                ),
            )

        return output_path.resolve()


def _bbox_center(bbox):
    return np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        dtype=np.float32,
    )


def _column_name(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell_xml(row, column, value, style=0):
    reference = f"{_column_name(column)}{row}"
    style_attribute = f' s="{style}"' if style else ""

    if value == "":
        return f'<c r="{reference}"{style_attribute}/>'
    if isinstance(value, (int, float)):
        return f'<c r="{reference}"{style_attribute}><v>{value}</v></c>'

    text = escape(str(value))
    return (
        f'<c r="{reference}" t="inlineStr"{style_attribute}>'
        f"<is><t>{text}</t></is></c>"
    )


def _worksheet_xml(rows, widths):
    last_column = _column_name(len(widths))
    last_row = max(1, len(rows))
    column_xml = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )

    row_xml = []
    for row_index, values in enumerate(rows, start=1):
        style = 1 if row_index == 1 else 0
        height = ' ht="24" customHeight="1"' if row_index == 1 else ""
        cells = "".join(
            _cell_xml(row_index, column_index, value, style)
            for column_index, value in enumerate(values, start=1)
        )
        row_xml.append(f'<row r="{row_index}"{height}>{cells}</row>')

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:{last_column}{last_row}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols>{column_xml}</cols>
  <sheetData>{''.join(row_xml)}</sheetData>
  <autoFilter ref="A1:{last_column}{last_row}"/>
</worksheet>"""


def _content_types_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def _root_relationships_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _workbook_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="逐幀摘要" sheetId="1" r:id="rId1"/>
    <sheet name="座標明細" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>"""


def _workbook_relationships_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Arial"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Arial"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center"/>
    </xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _core_properties_xml():
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Bat Monitor</dc:creator>
  <cp:lastModifiedBy>Bat Monitor</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>"""


def _app_properties_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Bat Monitor</Application>
</Properties>"""
