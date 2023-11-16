import face_recognition as fr

known = fr.load_image_file(known_path)
known_encoding = fr.face_encodings(known)

unknown = fr.load_image_file(unkwown_path)
unknown_encoding = fr.face_encodings(unknown)


