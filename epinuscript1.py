import sqlite3
from ultralytics import YOLO

#load in pretrained model for inference testing
def perform_object_detection(image_path):
    model_path = "Downloads/best-yolov5.pt" #old model
    
    yolo = YOLO(model_path)

    results = yolo(image_path)

    return results

# very basic sql query to get data using food_id
def simple_query(class_label):
    conn = sqlite3.connect('Downloads/EpiNu_New.db') #this might be outdated
    
    c = conn.cursor()
    
    c.execute("SELECT * FROM nutrient_levels WHERE food_id=?", (class_label,)) #using *food_id* for testing
    
    return c.fetchall()

# theres lots of other data from the detection boxes, this function just gets the class & confidence from output object
def extract_class(result):
    class_tensor = result[0].boxes.cls
    conf_tensor = result[0].boxes.conf
    class_index = int(class_tensor.item())
    confidence = float(conf_tensor.item())
    detected_class = result[0].names[class_index]

    return detected_class, confidence


def main(image_path):
    # object detection
    yolo_output = perform_object_detection(image_path)

    # extract basic data
    detected_class = extract_class(yolo_output)

    # execute basic sql query
    nutrient_info = simple_query('2009') ##yolo output labels dont match with database yet, using a random id for testing
    #nutrient_info = simple_query(detected_class)

    print("Detected Food Class:", detected_class[0])
    print("Confidence:", detected_class[1])
    print("Nutrient Information:", nutrient_info)
    print("Detection Box Data:", yolo_output[0].boxes)

if __name__ == "__main__":
    image_path = "Desktop/databases/image test/banana1.jpeg" #random webscrapped image for testing
    main(image_path)