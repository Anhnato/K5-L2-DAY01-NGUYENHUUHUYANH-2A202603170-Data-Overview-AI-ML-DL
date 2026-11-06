# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python3

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
}

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh. Lý do là vì mô hình YOLO nhận toàn bộ hình ảnh làm đầu vào và sử dụng `result.probs.top5` để lấy 5 nhãn phân loại có xác suất cao nhất cho hình ảnh đó.
- Mỗi Record lưu thông tin về một nhãn dán dự đoán, bao gồm `class_id`, `class_name`, `rank` và `scor`, đồng thời liên kết hình ảnh thôn qua `sample_id` và `coco_image_id`.
- Code không thực hiện phát hiện vị trí vật thể bẳng Bounding Box hoặc phân loại riêng từng vùng ảnh nên đây không phải Object-level classification.



3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm taxonomy/checkpoint chứ không phải model tự nghĩ ra, class list đến từ `classification_model = YOLO(CLASSIFICATION_MODEL_FILE)` và `names = result.names` cung cấp danh sách tên lớp tương ứng với ID của chúng.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cả 3 trường cung cấp 3 loại thông tin khác nhau, giúp Record có thể được định danh chính xác, kiểm tra, đối chiếu và sử dụng lại trong các hệ thống Data Annotation hoặc Machine Learning.
- `class_id` có vai trò định danh lớp bằng số.
- `class_name` vai trò tên dễ đọc của lớp.
- `taxonomy_name` xác định hệ thống phân loại.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Nếu trong ảnh có nhieeuf vật thể hoặc nhiều chủ đề, guideline cần quy định rõ quy tắc chọn lớp đại diện (Primary Label / Dominant Class) để tất cả annotator có cùng 1 cách đánh giá và chỉ chọn 1 lớp duy nhất.
- Có 5 nội dung quan trọng:
    - Main subject: Vật thể hoặc chủ đề chính của ảnh là gì.
    - Dominance: Nếu có nhiều vật thể, vật thể nào là chiếm ưu thế.
    - Visual prominence: Có xét kích thước, vị trí và độ nổi bật của vật thể không.
    - Context: Bối cảnh của vật thể.
    - Tie-breaking: Nếu 2 hoặc nhiều chủ đề ngang nhau thì chọn lớp nào.

6. Vì sao không được lấy `score` của model làm ground truth?
- Vì `score` chỉ thể hiện mức độ tin tưởng của model đối với dự đoán của nó, chứ không chứng minh dự đoán đó đúng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 5,
    "class_name": "bus",
    "score": 0.912557,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
  },

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên trái là giá trị `x_min` và `y_min` còn góc dưới phải là giá trị `x_max` và `y_max` từ `bbox_xyxy`.
- Chiều rộng là giá trị `round(x_max_raw-x_min_raw, 2)` từ `bbox_width`.
- Chiều dài là giá trị `round(y_max_raw-y_min_raw, 2)` từ `bbox_height`.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Threshold: 0.2 | Total prediction: 80.
- Threshold: 0.6 | Total prediction: 25.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ Confidence Threshold, model thường dữ lại nhiều Bounding Box hươn. Vì vậy, Reviewer phải dành thêm thời gian để kiểm tra, xác minh và loại bỏ những prediction không chính xác.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Quy tắc gán nhãn Bounding Box: người gán nhãn phải vẽ 1 Bounding Box bao quanh toàn bộ phần nhìn thấy được của đối tượng cần nhận diện, sử dụng hình chữ nhật nhỏ nhất có thể nhưng không cắt mất bất kỳ phần nào của đối tượng. Box phải bám sát đường viền đối tượng, hạn chế tối đa vùng nền không thuộc đối tượng. Nếu đối tượng bị che khuất một phần, chỉ vẽ box bao quanh phần thực sự nhìn thấy được, không tự suy đoán hoặc mở rộng box để bao gồm phần bị che khuất. Mỗi đối tượng riêng biệt phải có một bounding box riêng.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Nên đưa lên hỏi Lab Coach trước, thay vì tự ý bổ sung quy tắc vào guideline. Vì nếu tự thêm quy tắc, annotator có thể áp dựng sai so với tiêu chuẩn chung của dự án.
-Chưa biết dự án có quy định cụ thể về vật thể bị che khuất hoặc nằm ngoài khung hình hay chưa.
- Lab Coach có thể xác nhận cách xử lý và quyết định liệu đây có phải trường hợp cần cập nhật guideline hay chỉ là tình huống đặc biệt.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
      [
        141.0,
        193.0
      ],
      [
        139.0,
        193.0
      ],
      [
        137.0,
        195.0
      ],
      [
        136.0,
        195.0
      ],
      [
        135.0,
        196.0
      ],
      [
        134.0,
        196.0
      ],
      [
        133.0,
        197.0
      ],
      [
        132.0,
        197.0
      ],
      [
        131.0,
        198.0
      ],
      [
        130.0,
        198.0
      ],
      [
        128.0,
        200.0
      ],
      [
        127.0,
        200.0
      ],
      [
        126.0,
        201.0
      ],
      [
        125.0,
        201.0
      ],
      [
        123.0,
        203.0
      ],
      [
        122.0,
        203.0
      ],
      [
        121.0,
        204.0
      ],
      [
        120.0,
        204.0
      ],
      [
        119.0,
        205.0
      ],
      [
        117.0,
        205.0
      ],
      [
        116.0,
        206.0
      ],
      [
        115.0,
        206.0
      ],
      [
        114.0,
        207.0
      ],
      [
        113.0,
        207.0
      ],
      [
        111.0,
        209.0
      ],
      [
        110.0,
        209.0
      ],
      [
        109.0,
        210.0
      ],
      [
        108.0,
        210.0
      ],
      [
        105.0,
        213.0
      ],
      [
        104.0,
        213.0
      ],
      [
        98.0,
        219.0
      ],
      [
        98.0,
        220.0
      ],
      [
        96.0,
        222.0
      ],
      [
        96.0,
        312.0
      ],
      [
        97.0,
        313.0
      ],
      [
        97.0,
        314.0
      ],
      [
        98.0,
        315.0
      ],
      [
        101.0,
        315.0
      ],
      [
        102.0,
        316.0
      ],
      [
        104.0,
        316.0
      ],
      [
        105.0,
        317.0
      ],
      [
        106.0,
        317.0
      ],
      [
        107.0,
        318.0
      ],
      [
        108.0,
        318.0
      ],
      [
        109.0,
        319.0
      ],
      [
        113.0,
        319.0
      ],
      [
        114.0,
        318.0
      ],
      [
        116.0,
        318.0
      ],
      [
        117.0,
        317.0
      ],
      [
        122.0,
        317.0
      ],
      [
        123.0,
        316.0
      ],
      [
        131.0,
        316.0
      ],
      [
        132.0,
        315.0
      ],
      [
        163.0,
        315.0
      ],
      [
        164.0,
        314.0
      ],
      [
        170.0,
        314.0
      ],
      [
        171.0,
        315.0
      ],
      [
        176.0,
        315.0
      ],
      [
        177.0,
        316.0
      ],
      [
        183.0,
        316.0
      ],
      [
        184.0,
        315.0
      ],
      [
        185.0,
        315.0
      ],
      [
        188.0,
        312.0
      ],
      [
        188.0,
        311.0
      ],
      [
        190.0,
        309.0
      ],
      [
        190.0,
        308.0
      ],
      [
        197.0,
        301.0
      ],
      [
        198.0,
        301.0
      ],
      [
        199.0,
        300.0
      ],
      [
        201.0,
        300.0
      ],
      [
        202.0,
        299.0
      ],
      [
        203.0,
        299.0
      ],
      [
        204.0,
        298.0
      ],
      [
        205.0,
        298.0
      ],
      [
        206.0,
        297.0
      ],
      [
        209.0,
        297.0
      ],
      [
        210.0,
        296.0
      ],
      [
        211.0,
        296.0
      ],
      [
        213.0,
        294.0
      ],
      [
        213.0,
        293.0
      ],
      [
        214.0,
        292.0
      ],
      [
        214.0,
        291.0
      ],
      [
        215.0,
        290.0
      ],
      [
        215.0,
        289.0
      ],
      [
        217.0,
        287.0
      ],
      [
        217.0,
        286.0
      ],
      [
        219.0,
        284.0
      ],
      [
        219.0,
        283.0
      ],
      [
        220.0,
        282.0
      ],
      [
        220.0,
        281.0
      ],
      [
        221.0,
        280.0
      ],
      [
        221.0,
        279.0
      ],
      [
        222.0,
        278.0
      ],
      [
        222.0,
        276.0
      ],
      [
        223.0,
        275.0
      ],
      [
        223.0,
        234.0
      ],
      [
        222.0,
        233.0
      ],
      [
        222.0,
        226.0
      ],
      [
        221.0,
        225.0
      ],
      [
        221.0,
        220.0
      ],
      [
        220.0,
        219.0
      ],
      [
        220.0,
        208.0
      ],
      [
        219.0,
        207.0
      ],
      [
        219.0,
        202.0
      ],
      [
        218.0,
        201.0
      ],
      [
        218.0,
        198.0
      ],
      [
        217.0,
        197.0
      ],
      [
        217.0,
        196.0
      ],
      [
        213.0,
        192.0
      ],
      [
        211.0,
        192.0
      ],
      [
        210.0,
        191.0
      ],
      [
        206.0,
        191.0
      ],
      [
        205.0,
        190.0
      ],
      [
        199.0,
        190.0
      ],
      [
        198.0,
        189.0
      ]
    ]
  }

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cung cấp thông tin về đường biên và hình dạng thực tế của từng vật thể, trong khi bounding box chỉ cung cấp vùng hình chữ nhật bao quanh vật thể.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` dùng để định danh duy nhất từng đối tượng được phát hiện trong ảnh.
- Bao gồm `sample_id`: ID ảnh của đối tượng; `index + 1` số thứ tự của đối tượng trong ảnh; `:03d` định dạng số thành 3 chữ số, thêm số 0 ở đầu nếu cần.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Người gán nhãn phải vẽ đường biên mask bám sát đường viền của từng đối tượng, bao phủ toàn bộ phần đối tượng nhìn thấy được và không bao gồm các vùng background hoặc đối tượng khác. Mỗi đối tượng riêng biệt phải có một mask riêng, kể cả khi các đối tượng cùng class hoặc chồng lấn lên nhau. Đối với phần bị che khuất, chỉ vẽ mask cho phần thực sự nhìn thấy được, không tự suy đoán hình dạng của phần bị che.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Nên đưa lên hỏi Lab Coach trước, thay vì tự ý bổ sung quy tắc vào guideline. Vì nếu tự thêm quy tắc, annotator có thể áp dựng sai so với tiêu chuẩn chung của dự án.
-Chưa biết dự án có quy định cụ thể về vật thể bị che khuất hoặc nằm ngoài khung hình hay chưa.
- Lab Coach có thể xác nhận cách xử lý và quyết định liệu đây có phải trường hợp cần cập nhật guideline hay chỉ là tình huống đặc biệt.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Ground truth là nhãn chính xác được con người xác định cho toàn bộ ảnh, thông qua quy trình annotation và review theo guideline |  | Annotator chịu trách nhiệm sác định nhãn cho ảnh theo guideline. | Reviewer đánh giá nhãn có đúng và nhất quán với guideline không. |
| Phát hiện vật thể | Là tập hợp các bounding box và class label được con người gán và xác minh cho từng vật thể xuất hiện trong ảnh. |  | Annotator có nhiệm vụ xác định tất cả đối tượng thuộc các class được yêu cầu và vẽ bounding box cho từng đối tượng | Reviewer phải kiểm tra cả độ chính xác của Bounding Box và tính đầy đủ của dữ liệu. |
| Instance segmentation | là tập hợp các mask riêng biệt cho từng vật thể, đi kèm class label và thông tin định danh instance. |  | Annotator chịu trách nhiệm tạo mask cho từng đối tượng thuộc class được yêu cầu. | Reviewer phải kiểm tra không chỉ class và vị trí mà còn cả chất lượng đường biên của mask. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Toàn bộ dữ liệu hình ảnh, nhãn Ground truth và kết quả dự đoán từ mô hình phải được bảo mật, chỉ sử dụng cho mục đích dự án và chỉ người được phân quyền mới được phép truy cập, chỉnh sử hoặc chia sẻ.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ báo cáo cho Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
