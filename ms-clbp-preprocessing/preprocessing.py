from multiprocessing import Pool, cpu_count

def preprocess(dataset_dir, target_path):
    for cls_num, cls in enumerate(classes):
        arguments = []
        class_dir = os.path.join(dataset_dir, cls)
        for filename in sorted(os.listdir(class_dir)):
            filepath = os.path.join(class_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img.shape != (256, 256):
                img = cv2.resize(img, (256, 256))
            img = normalize(img)
            arguments.append((img, scales, n_points, radii, patch_size))
        with Pool(cpu_count()) as p:
            feature_vecs = p.starmap(ms_clbp_feature_matrices, arguments)
        with open(os.path.join(os.getcwd(), target_path, f"{cls}_feature_mats.pkl"), 'wb') as f:
            pickle.dump(feature_vecs, f)
