
# CELDA 4
# Contamos cantidad de usuarios y filtramos aquellos con menos de 200 valoraciones
user_counts = df_ratings['user'].value_counts()
df_ratings = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 200].index)]

# Hacemos lo mismo que arriba pero con los libros 
book_counts = df_ratings['isbn'].value_counts()
df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

# CELDA 5 
# Creacion de matriz y conversión a matriz dispersa
user_book_matrix = df_ratings.pivot(index='user', columns='isbn', values='rating').fillna(0)
user_book_sparse_matrix = csr_matrix(user_book_matrix.values)


# CELDA 6

#Modelo KKN

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_sparse_matrix.T)

# este tira error, revisar copilot
