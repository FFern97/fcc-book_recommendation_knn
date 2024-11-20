
# CELDA 4
# Filtrar usuarios con menos de 200 valoraciones
user_counts = df_ratings['user'].value_counts()
df_ratings_filtered_users = df_ratings[~df_ratings['user'].isin(user_counts[user_counts < 200].index)]

# Filtrar libros con menos de 100 valoraciones después del filtrado de usuarios
isbn_counts = df_ratings_filtered_users['isbn'].value_counts()
df_ratings_filtered_books = df_ratings_filtered_users[~df_ratings_filtered_users['isbn'].isin(isbn_counts[isbn_counts < 100].index)]

# Verificar si el libro "Where the Heart Is (Oprah's Book Club (Paperback))" está después del filtrado
book_title_test = "Where the Heart Is (Oprah's Book Club (Paperback))"
book_isbn_test = df_books[df_books['title'] == book_title_test]['isbn'].values[0]
isbn_in_final_filtered_test = book_isbn_test in df_ratings_filtered_books['isbn'].values
print(f"¿El ISBN '{book_isbn_test}' está después del ajuste final? {isbn_in_final_filtered_test}")

# CELDA 5 
# Crear una matriz de usuario-libro
user_book_matrix = df_ratings_filtered_books.pivot(index='user', columns='isbn', values='rating').fillna(0)

# Convertir a una matriz dispersa
user_book_sparse_matrix = csr_matrix(user_book_matrix.values)
print(f"Tamaño de la matriz de usuario-libro: {user_book_matrix.shape}")

# Crear el modelo de vecinos más cercanos
if user_book_matrix.shape[0] > 0 and user_book_matrix.shape[1] > 0:
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_book_sparse_matrix.T)  # Transponer para tener libros como filas
else:
    print("La matriz de usuario-libro está vacía después del filtrado.")

