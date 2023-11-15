import { assert, log } from 'console'
import fs from 'fs'
import puppeteer from 'puppeteer'

const DOWNLOAD_PATH = 'repositories.csv'

function appendToCSV(data: string) {
    assert(fs.existsSync(DOWNLOAD_PATH))
    const file = fs.readFileSync(DOWNLOAD_PATH, 'utf8')
    if (file.includes(data)) {
        return false
    }

    fs.appendFileSync(DOWNLOAD_PATH, data)
    return true
}

const main = async () => {
    console.clear()

    // get user arg
    const args = process.argv.slice(2)
    assert(args.length === 1, 'only one argument is allowed')
    const URL = args[0]
    log(`URL: ${URL}`)

    // init browser
    const browser = await puppeteer.launch({
        // slowMo: 1000,
        headless: 'new',
    })
    const page = await browser.newPage()
    await page.goto(URL)

    while (true) {
        // get links
        const links = await page.$$eval('article div:nth-child(1) a:nth-child(2)', (anchors) => {
            const hrefs = anchors.map((anchor) => anchor.getAttribute('href'))
            const url = hrefs.map((href) => `https://github.com${href}`)
            return url
        })
        let numNewLinks = 0
        links.forEach((link) => {
            if (appendToCSV(`${link}\n`)) {
                numNewLinks++
            }
        })
        log(`found ${numNewLinks} new links`)

        // get further links
        const path = '.ajax-pagination-btn'
        await page.waitForSelector(path, { timeout: 100_000_000 })
        const nextPageButton = await page.$$(path)
        const nextPageButtonDisabled = await nextPageButton[0].evaluate((node) => node.disabled)
        if (nextPageButtonDisabled) {
            const window = await page.evaluateHandle(() => window)
            await page.evaluate((window) => window.scrollY, window)
            continue
        } else {
            log('reached bottom, clicking next page')
            await nextPageButton[0].click()
            await page.waitForTimeout(1000)
        }
    }
}
main()
