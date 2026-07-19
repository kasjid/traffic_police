from ultralytics import YOLO

model = YOLO("/2024212456/workspace/traffic_police/yolo26/models/baseline/logs/coco_baseline/coco_baseline/weights/best.pt")

results = model.train(                                                                                                                                                  
      data="datasets/traffic_class1.yaml",
      imgsz=640,                                                                                                                                                          
                                                                                                                                                                          
      # 核心参数（针对1200张数据优化）                                                                                                                                    
      epochs=150,                                                                                                                                                         
      batch=32,             # 小batch，更多更新                                                                                                                           
                                                                                                                                                                          
      device=1,                                                                                                                                                           
                                                                                                                                                                          
      # 学习率                                                                                                                                                            
      lr0=0.001,  
      lrf=0.01,                                                                                                                                                           
      optimizer="AdamW",                                                                                                                                                  
      cos_lr=True,                                                                                                                                                        
      warmup_epochs=5,                                                                                                                                                    
                                                                                                                                                                          
      # 早停                                                                                                                                                              
      patience=50,                                                                                                                                                        
                                                                                                                                                                          
      # 强数据增强（小数据集必须）                                                                                                                                        
      mosaic=1.0,                                                                                                                                                         
      mixup=0.15,                                                                                                                                                         
      copy_paste=0.1,                                                                                                                                                     
                                                                                                                                                                          
      project="/2024212456/workspace/traffic_police/yolo26/models/baseline/logs/traffic_class1_tune",                                                                      
      name="traffic_class1_baseline_by_coco_bs32_epoch150",                                                                                                                               
  ) 
