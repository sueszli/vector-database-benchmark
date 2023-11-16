def multi_genexpr(blog_posts):
    if False:
        while True:
            i = 10
    return (entry for blog_post in blog_posts for entry in blog_post.entry_set)