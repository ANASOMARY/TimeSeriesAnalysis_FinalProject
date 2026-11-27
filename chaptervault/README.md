# ChapterVault

A private reading room for unpublished manuscripts. You upload only the chapters you want
reviewed, decide which reviewer may read which chapter and for how long, and take access away
again whenever you like. Reviewers read in their browser; the file itself never leaves the server.

| The reviewer's reader | The author's console |
|---|---|
| ![Watermarked reader with a notes panel](docs/reader.jpg) | ![Chapter page listing who can read it](docs/author-console.jpg) |

## What it does

| | |
|---|---|
| **Upload selected chapters** | PDF, Word (`.docx`, `.doc`), OpenDocument, RTF, EPUB or plain text, several files at once. Uploading the whole book? Give a page range (`45-78`) and only those pages are kept. |
| **Encrypted storage** | Every chapter is encrypted with AES-256-GCM before it touches the disk. |
| **No file download** | Reviewers receive pages as images rendered on the server. There is no PDF to save, forward or open elsewhere. |
| **Personal watermark** | Each page is stamped with text you write, by default the reviewer's name, email address, the time and their IP address, so a leaked screenshot points at its source. |
| **Per-reviewer rules** | For each reviewer and chapter: a start date, an end date, a limit on the number of opens, and whether they may leave notes. |
| **Instant revocation** | Revoke one grant, suspend a chapter or suspend a reviewer. An open reader closes within about 20 seconds. |
| **Device locking** | An account works on a limited number of devices (2 by default) and only one session at a time, so a login cannot be passed around. |
| **Printing and copying blocked** | Printing produces a blank page; text selection, right-click, save and copy are disabled; the page is hidden when the window loses focus. |
| **Audit trail** | Sign-ins, every chapter opened, every page viewed, reading time, and each blocked print or copy attempt. Exportable as a spreadsheet. |
| **Reviewer feedback** | Reviewers click a paragraph, or drag a box around any passage, and pin a note to it (or note a whole page), and can edit or delete their own notes. You see each note with the passage quoted beside it, can jump to it on the page, and can export everything. |
| **No paragraph skipped** | The reader tracks which paragraphs each reviewer has actually had in front of them, flags any they scrolled past, and only lets a chapter be marked as reviewed when none are left. You can see exactly what each reviewer has not covered. |
| **Reviewer's own summary** | Each reviewer sees where they stand: share read, whether their pace reaches the target date, and a map of every page showing what is read, partly read or untouched. |
| **Progress report** | A page that shows how far each reviewer has got, their pace, and whether they will finish by your target date, with a chart of progress against that target. Prints as a two-page report or exports as a spreadsheet. |
| **Note categories** | You define the kinds of note ("Sounds AI-written", "Hard to follow", …), each with a one-line explanation. Reviewers tap any that apply, with or without typed details. Feedback is counted and filterable by category. |
| **Invitations by email** | Enter a mail account once and each reviewer is emailed their personal invitation as soon as you add them. Resend it, or email a password-reset link, from the reviewer's page. Without a mail account you copy the link and send it yourself. |
| **Confidentiality terms** | Reviewers must accept your terms before they can open anything (editable, optional). |

## Run it on your computer

Requires Python 3.10 or newer.

```bash
cd chaptervault
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

The first start prints a one-time link:

```
No author account exists yet. Open this one-time link to create it:
  http://localhost:8000/setup?token=…
```

Open it, create your author account, and sign in.

Word files convert with their original layout if [LibreOffice](https://www.libreoffice.org) is
installed. Without it, `.docx` files still convert, but the text is re-flowed onto plain A4 pages;
exporting to PDF from Word first gives you exact control over what reviewers see.

## Using it

1. **Chapters → Upload chapters.** Pick the files. Each is converted, encrypted and listed.
   *Preview as a reviewer sees it* shows the watermarked result.
2. **Reviewers → Add a reviewer.** Enter a name and email and tick the chapters to share. If email
   is set up (below), their invitation is sent straight away; otherwise you get a personal link to
   send yourself (*Copy link* or *Write email*). The reviewer uses it once to choose a password.
3. **Access.** A grid of reviewers against chapters: tick to grant, untick to revoke. Open a chapter
   or a reviewer to fine-tune the dates and open limits of a single grant.
4. **Activity.** See who read what, how far they got, and anything that looked like a copy attempt.

### Progress report

**Progress** is built for the start of a weekly call. Set the target date at the top of the page.

- **Headline figures:** the share of assigned pages read, pages read in the last seven days compared
  with the seven before, how many reviewers are on course, and notes written.
- **The chart** sets pages read so far against an even pace to the target, and shows where the
  current pace leads. That forecast adds up each reviewer's own pace, so it bends as people finish
  and levels off short of 100% if anyone is not reading.
- **Reviewers,** those needing attention first: progress, the last seven days, pace against the pace
  needed, projected finish date, a status, and a small trajectory. **Chapters,** least-read first.
- **A warning** when a reviewer's access to an unfinished chapter ends before the target date.
- **Print or save as PDF** gives a landscape report (summary and chart, then the tables).
  **Download data** gives the reviewer table as a spreadsheet.

How it is worked out:

- A page counts as *read* once every paragraph on it has been covered (see *Paragraph coverage*
  below). Text that was only loaded, or scrolled past, does not count.
- *Pace* is pages a day over the last 14 days, or since the chapters were assigned if that is more
  recent. Days without reading count, so going quiet lowers it.
- *Projected finish* is today plus the pages left divided by that pace. The whole review's forecast
  is the date its last reviewer finishes.
- *On track* means the pace meets what is needed to finish by the target; *at risk*, at least 60% of
  it; *behind*, less than that, including no recent reading.

### Paragraph coverage

The reader works out where each paragraph sits on the page and follows the reviewer through them.

- A paragraph is **covered** when the text on screen has been visible, with the window in focus,
  for at least as long as a very fast reader would need (600 words a minute); when the reviewer
  ticks it by hand; or when they pin a note to it. Time on screen is shared among the visible
  paragraphs by length, so a full screen of text takes as long as its word count says.
- Each paragraph not yet covered has a small marker in the page margin. One that was on screen and
  left unread turns amber. Clicking a marker ticks that paragraph as reviewed.
- The toolbar shows how many are left and jumps to the next one. When none are left it becomes
  **Mark as reviewed**; a chapter cannot be marked while any paragraph is uncovered.
- Very short blocks (page numbers, running heads, most headings) are not required. A page with no
  text, such as a figure or a scan, counts once it has been on screen for five seconds.
- On a chapter's page, each reviewer's row shows paragraphs covered and whether they marked it as
  reviewed, and links to the list of paragraphs they have not covered, with how the rest were
  covered (read on screen, commented on, or ticked by hand).
- Replacing a chapter's file starts reading again for that chapter: its paragraphs have changed.

This shows that every paragraph was put in front of the reviewer and acknowledged. It cannot show
that it was read with care; a tick is an attestation.

Reviewers see the result on their library page: the share of their pages read, a sentence on whether
their pace reaches the target date and what pace would, their trajectory, and one cell per page
(read, partly read, not read, with a dot where they left notes) that opens the chapter at that page.

### Note categories

**Categories** holds the kinds of note reviewers can choose from. A vault starts with seven
(*Sounds AI-written*, *Hard to follow*, *Awkward wording*, *Filler*, *Repetitive*, *Eyebrow text*,
*Works well*); add, rename, recolour, reorder or remove them as you like.

- In the reader, the categories appear as buttons above the note box. A reviewer can pick several,
  pick one and save without typing, or type a note with no category at all.
- The explanation of a category shows under the buttons when the reviewer points at it, moves to it
  with the keyboard, or picks it, so it is visible on phones and tablets too.
- A note's highlight on the page takes the colour of its first category.
- Each chapter's feedback shows a count per category, filters by it, and exports it as a column.
- Retiring a category hides it from reviewers but keeps it on notes already written. A category
  that has been used is retired rather than deleted.

If a reviewer changes computer or forgets their password, open their page and use
*Reset devices*, *Email a password reset link* or *Create password reset link*.

### Sending invitations by email

Under **Settings → Email**, enter a mail account. Any standard SMTP service works, the connection is
always encrypted, and the password is stored encrypted and never shown again. Save, then use
*Send a test email* at the foot of the page.

| | Mail server | Port | Connection | Username and password |
|---|---|---|---|---|
| Gmail or Google Workspace | `smtp.gmail.com` | 587 | STARTTLS | Your address, and an *app password* (needs 2-Step Verification turned on) |
| Amazon SES | `email-smtp.<region>.amazonaws.com` | 587 | STARTTLS | *SMTP credentials* created in the SES console |
| Resend | `smtp.resend.com` | 465 | SSL/TLS | Username `resend`, password a Resend API key with sending access |
| Another provider | Look up its "SMTP settings" | 587 or 465 | STARTTLS or SSL/TLS | Usually your mailbox sign-in or an app password |

If you send from a dedicated sending domain that has no inbox (common with services such as
Resend, where a subdomain like `mail.example.org` is verified), fill in *Replies go to* with a real
mailbox so that a reviewer's reply still reaches you.

Notes on Amazon SES: verify the sender address or domain first, and be aware that a new SES account
is in a *sandbox* that only delivers to addresses you have verified until AWS approves a request
for production access.

The subject and wording of the invitation are yours to edit in the same section. `{name}`, `{link}`,
`{book}`, `{author}` and `{expires}` are filled in for each reviewer. Resending an invitation reuses
the link already sent while it is still valid, and issues a fresh one once it has expired.

## Putting it online for reviewers

Reviewers need to reach the server over **HTTPS**. Do not expose it over plain HTTP: passwords and
pages would travel unencrypted.

### On AWS (one command)

`deploy/aws/chaptervault.yaml` is a CloudFormation template that creates everything: one small
server with a fixed IP address, a firewall open only on the web ports, an encrypted disk, an
automatic HTTPS certificate and daily backups kept for 14 days. Expect roughly $20 a month.

1. Sign in to the AWS console and pick a region (top right) near you and your reviewers.
2. Open **CloudShell** (the terminal icon in the top bar) and paste:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/ANASOMARY/TimeSeriesAnalysis_FinalProject/ccr-ad86fbe7-utaqxg/chaptervault/deploy/aws/chaptervault.yaml -o chaptervault.yaml \
   && aws cloudformation deploy --stack-name chaptervault --template-file chaptervault.yaml --capabilities CAPABILITY_IAM \
   && aws cloudformation describe-stacks --stack-name chaptervault --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output text
   ```

3. Wait about ten minutes. It ends by printing three lines:
   - `Step1CreateYourAccount`: open the link between the quotes once, to create your author account.
   - `WebAddress`: the address you and your reviewers use.
   - `ServerIpAddress`: only needed if you bring your own domain (below).

Without further options you get an automatic address of the form `https://1-2-3-4.sslip.io`.
It works immediately, but it relies on a free third-party DNS service, so treat it as a way to
try things out. **Before inviting reviewers, use an address of your own**: add this to the
`deploy` line.

```bash
--parameter-overrides DomainName=review.example.com HostedZoneId=Z0123456789ABC
```

`HostedZoneId` is for domains whose DNS is in Route 53 (find it under Route 53 → Hosted zones); the
DNS record is then created for you. If your DNS is elsewhere, leave it out and create an `A` record
for your address pointing to `ServerIpAddress`.

If the command fails, this shows why; after a failed first attempt, delete the stack before retrying.

```bash
aws cloudformation describe-stack-events --stack-name chaptervault \
  --query "StackEvents[?ResourceStatus=='CREATE_FAILED'].[LogicalResourceId,ResourceStatusReason]" --output text
aws cloudformation delete-stack --stack-name chaptervault
```

Later on:

- **Update to the latest code.** In CloudShell, in the same region:

  ```bash
  curl -fsSL https://raw.githubusercontent.com/ANASOMARY/TimeSeriesAnalysis_FinalProject/ccr-ad86fbe7-utaqxg/chaptervault/deploy/aws/update.sh | bash
  ```

  Chapters, accounts, notes and settings are kept. The same thing by hand: in the EC2 console
  select the server, choose *Connect → Session Manager*, and run
  `sudo bash /opt/chaptervault/src/chaptervault/deploy/aws/deploy.sh`.
- **Move to an address of your own.** Create a DNS `A` record for the new name (for example
  `chaptervault.example.org`) pointing at `ServerIpAddress`, wait a few minutes, then in CloudShell:

  ```bash
  curl -fsSL https://raw.githubusercontent.com/ANASOMARY/TimeSeriesAnalysis_FinalProject/ccr-ad86fbe7-utaqxg/chaptervault/deploy/aws/set-address.sh | bash -s -- chaptervault.example.org
  ```

  It checks the DNS record first, moves the site, and waits for the new certificate. The old address
  stops working, so do this before inviting reviewers: links already sent must be resent, and
  everyone signs in again at the new address (a reviewer's browser counts as a new device there).
- **Remove everything.** `aws cloudformation delete-stack --stack-name chaptervault` deletes the
  server and every chapter on it.

### On any other server

With Docker and a domain name pointing at the server:

```bash
cd chaptervault
DOMAIN=review.example.com docker compose up -d --build
docker compose logs app      # shows the one-time setup link
```

This starts ChapterVault behind [Caddy](https://caddyserver.com), which obtains and renews the HTTPS
certificate automatically. The image includes LibreOffice for Word conversion.

Or run `python run.py --host 0.0.0.0` behind any reverse proxy that terminates TLS, and set
`CHAPTERVAULT_BEHIND_PROXY=1` and `CHAPTERVAULT_BASE_URL`.

## Settings

| Environment variable | Purpose | Default |
|---|---|---|
| `CHAPTERVAULT_DATA` | Folder for the database, encrypted chapters and key | `./data` |
| `CHAPTERVAULT_BASE_URL` | Public address, used in invite links | taken from the request |
| `CHAPTERVAULT_BEHIND_PROXY` | Set to `1` behind a TLS-terminating proxy, so HTTPS and client IPs are detected | off |
| `CHAPTERVAULT_MASTER_KEY` | 32-byte key (url-safe base64). Keeps the key out of the data folder | generated in `data/master.key` |
| `CHAPTERVAULT_ADMIN_EMAIL`, `CHAPTERVAULT_ADMIN_PASSWORD`, `CHAPTERVAULT_ADMIN_NAME` | Create the author account at first start instead of using the setup link | unset |
| `CHAPTERVAULT_MAX_UPLOAD_MB` | Largest accepted upload | `200` |
| `CHAPTERVAULT_SETUP_TOKEN` | Fixes the one-time setup token (32+ characters) so the setup link survives restarts | random per start |
| `PORT`, `CHAPTERVAULT_HOST` | Listening port and address | `8000`, `127.0.0.1` |

Defaults for new grants, the watermark and the confidentiality terms are edited in the app under
**Settings**. The watermark is two texts you write yourself: one repeated across the page (up to
three lines) and one line at the foot. Words in braces are filled in for each reader: `{name}`,
`{email}`, `{date}`, `{ip}`, `{ref}`, `{book}` and `{author}`. *Preview as a reviewer sees it* shows
the result with a stand-in reviewer.

## Backups

Back up the whole data folder. `master.key` is the only thing that can decrypt the chapters: lose
it and they are unreadable; leak it together with the data folder and they are readable. For a
stricter setup, supply the key through `CHAPTERVAULT_MASTER_KEY` so it is stored apart from the data.

## What it protects against, and what it cannot

It stops the ordinary ways a manuscript gets passed on: there is no file to forward, no working
print or save, no shared logins, and access ends when you say so.

It runs in a web browser, so it **cannot physically stop** someone photographing their screen or
using a screenshot tool. Nothing browser-based can; dedicated DRM products close that gap by making
readers install a locked-down viewer application. ChapterVault's answer is deterrence and
traceability: every page a reviewer sees carries their own name, email address, time and IP
address across the text, the page is hidden when the window loses focus, and the log shows exactly
which pages each person opened.

Other limits worth knowing:

- Reading requires an internet connection; there is no offline mode.
- Device locking uses a browser cookie. Clearing cookies makes a browser look like a new device,
  which counts against the reviewer's device limit rather than bypassing it.
- The database (names, emails, notes, the activity log) is not encrypted; only the chapters are.

## Tests

```bash
pip install pytest aiosmtpd
python -m pytest tests
```

`aiosmtpd` is a small local mail server used by one test to prove real encrypted delivery; that
test is skipped if it is not installed.

The suite drives the real HTTP routes: encryption at rest, grant isolation between reviewers,
expiry, open limits, revocation of an open reader, device locking, single-session enforcement,
lockout, CSRF, the author-only console, pinned notes, invitation email, note categories, the
progress forecast and paragraph coverage.

## Licence note

ChapterVault renders pages with [PyMuPDF](https://pymupdf.readthedocs.io), which is licensed under
the AGPL. That is fine for running your own copy; read its terms before offering a modified version
to others as a hosted service.
