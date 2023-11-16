# diary

## 1) scraping a million files of code

there are 2 ways to scrape github:

1. **using their api**

    the issue of using their api is how aggressively they rate limit you.

    - anonymous client rate limit: 60 requests / hour
    - authenticated client rate limit: 5000 requests / hour
    - enterprise client rate limit: 15000 requests / hour

    now, there are a bunch of clever ways to bypass this issue:

    - bandwidth throttling: https://github.com/tomasbasham/ratelimit
    - rotating ip addresses: https://github.com/Ge0rg3/requests-ip-rotator, https://0xn3va.gitbook.io/cheat-sheets/web-application/improper-rate-limits#using-proxy-or-vpn, https://www.zenrows.com/blog/web-scraping-rate-limit

    you can also switch client id's, spoof your location and figure out how their rate limiter works through trial and error.

    but after several attempts i figured out that even with my fastest scraper it'd take me days to get all the data i need, because of how sophisticated their ratelimit is.

2. **web scraping**

    web scraping github by simply setting a language as a search filter doesn't work, because they only limit the results to 5 pages: https://github.com/search?q=language%3AC&type=repositories

    you could use random entries to scrape these 5 pages on multiple iterations, but it'd still be very time consuming and lots of manual work.

   i found 2 alternative paths:

    - using the `system size:>100000` query to filter out by the largest repositories to get a huge number of files at once. i did this manually. i got to around 300 links before i gave up.
    - scraping the topic pages for c/c++ (ie. https://github.com/topics/c) to get access to thousands of repositories within the same page, simply by clicking the "load more..." button. this way you don't even get rate limited and can run multiple requests that build on top of eachother. i managed to scrape around 2000 repository links this way.

after collecting a bunch of links to repositories i merged and sorted the files: 2340 original lines, 186 removed, 2154 remaining.

i then did the same thing again, but this time with python code.
