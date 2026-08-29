import re
from models import Person, PersonAlias, Identity, Conversation, Memory, Source

def get_person_by_key(db, person_key):
    if not person_key:
        return None
    return db.query(Person).filter_by(person_key=str(person_key).strip()).first()

def get_person_by_alias(db, alias):
    alias = str(alias or "").strip()
    if not alias:
        return None
    row = db.query(PersonAlias).filter_by(alias=alias).first()
    return row.person if row else None

def get_person_by_identity(db, display_name):
    display_name = str(display_name or "").strip()
    if not display_name:
        return None
    identity = db.query(Identity).filter_by(display_name=display_name, platform="kakaotalk").first()
    return identity.person if identity else None

def make_person_key(db):
    rows = db.query(Person).filter(Person.person_key.like("person_%")).all()
    max_number = 0
    for person in rows:
        match = re.match(r"person_(\d+)$", person.person_key)
        if match:
            max_number = max(max_number, int(match.group(1)))
    return f"person_{max_number + 1:03d}"

def get_or_create_self_person(db, canonical_name=None):
    self_person = db.query(Person).filter_by(person_key="self").first()
    if not self_person:
        self_person = Person(
            person_key="self",
            canonical_name=canonical_name or "이태양",
            person_type="self",
            status="active",
            observed_in_chat=1,
            confirmed=1,
        )
        db.add(self_person)
        db.commit()
        db.refresh(self_person)
    else:
        self_person.person_type = "self"
        self_person.status = "active"
        self_person.confirmed = 1
        self_person.observed_in_chat = 1
        if canonical_name:
            self_person.canonical_name = canonical_name
        db.commit()
    return self_person

def ensure_alias(db, person, alias, source=Source.DIRECT_STATEMENT, confidence=1.0):
    alias = str(alias or "").strip()
    if not alias:
        return
    existing = db.query(PersonAlias).filter_by(alias=alias).first()
    if existing:
        return
    db.add(PersonAlias(person_id=person.id, alias=alias, source=source, confidence=confidence))

def ensure_identity(db, person, display_name, is_primary=1):
    display_name = str(display_name or "").strip()
    if not display_name:
        return
    identity = db.query(Identity).filter_by(display_name=display_name, platform="kakaotalk").first()
    if identity:
        identity.person_id = person.id
        identity.target_key = person.person_key
        identity.platform = "kakaotalk"
        identity.is_primary = is_primary
        return
    db.add(Identity(
        person_id=person.id,
        target_key=person.person_key,
        platform="kakaotalk",
        display_name=display_name,
        is_primary=is_primary,
    ))

def merge_person_into_self(db, old_person, self_person, persona_id):
    if not old_person or old_person.id == self_person.id:
        return
    old_key = old_person.person_key
    print(f"[identity merge] {old_key} ({old_person.canonical_name}) -> self")

    for alias in db.query(PersonAlias).filter_by(person_id=old_person.id).all():
        existing = db.query(PersonAlias).filter_by(alias=alias.alias).first()
        if existing and existing.id != alias.id:
            if existing.person_id == self_person.id:
                db.delete(alias)
        else:
            alias.person_id = self_person.id

    for identity in db.query(Identity).filter_by(person_id=old_person.id).all():
        dup = db.query(Identity).filter(
            Identity.person_id == self_person.id,
            Identity.platform == identity.platform,
            Identity.display_name == identity.display_name,
            Identity.id != identity.id,
        ).first()
        if dup:
            db.delete(identity)
        else:
            identity.person_id = self_person.id
            identity.target_key = "self"

    try:
        conversations = db.query(Conversation).filter(Conversation.speaker_id == old_key).all()
        for conv in conversations:
            conv.speaker_id = "self"
            if not conv.speaker_name:
                conv.speaker_name = self_person.canonical_name
    except Exception as e:
        print(f"[identity merge] Conversation 정리 실패: {repr(e)}")

    try:
        memories = db.query(Memory).filter(Memory.persona_id == persona_id).all()
        for mem in memories:
            people = mem.people_involved or []
            if old_key in people:
                mem.people_involved = list(dict.fromkeys(["self" if str(v) == old_key else v for v in people]))
    except Exception as e:
        print(f"[identity merge] Memory 정리 실패: {repr(e)}")

    old_person.status = "merged"
    old_person.confirmed = 0
    old_person.notes = "merged_into=self"
    db.commit()

def cleanup_self_duplicates(db, sender, canonical_name, persona_id):
    self_person = get_or_create_self_person(db, canonical_name)
    candidates = []
    values = [str(sender or "").strip(), str(canonical_name or "").strip(), "self"]

    for val in values:
        if not val:
            continue
        for alias in db.query(PersonAlias).filter_by(alias=val).all():
            if alias.person and alias.person.id != self_person.id and alias.person not in candidates:
                candidates.append(alias.person)
        for identity in db.query(Identity).filter_by(display_name=val, platform="kakaotalk").all():
            if identity.person and identity.person.id != self_person.id and identity.person not in candidates:
                candidates.append(identity.person)
        for person in db.query(Person).filter(Person.canonical_name == val, Person.id != self_person.id, Person.status != "merged").all():
            if person not in candidates:
                candidates.append(person)

    for person in candidates:
        merge_person_into_self(db, person, self_person, persona_id)

    return self_person

def get_or_create_observed_person(db, display_name):
    display_name = str(display_name or "").strip()
    if not display_name:
        raise ValueError("display_name이 비어있음")

    identity = db.query(Identity).filter_by(display_name=display_name, platform="kakaotalk").first()
    if identity and identity.person:
        person = identity.person
        person.status = "active"
        person.observed_in_chat = 1
        if person.person_key == "self":
            person.person_type = "self"
            person.confirmed = 1
        db.commit()
        return person, False

    person = get_person_by_alias(db, display_name)
    if person:
        if person.status in ["inactive", "merged"]:
            person.status = "active"
        person.observed_in_chat = 1
        ensure_identity(db, person, display_name, is_primary=1)
        db.commit()
        return person, False

    person_key = make_person_key(db)
    person = Person(
        person_key=person_key,
        canonical_name=display_name,
        person_type="person",
        status="active",
        observed_in_chat=1,
        confirmed=0,
    )
    db.add(person)
    db.commit()
    db.refresh(person)

    ensure_alias(db, person, display_name, source=Source.OBSERVED, confidence=0.5)
    ensure_identity(db, person, display_name, is_primary=1)
    db.commit()
    return person, True

def extract_mentioned_people(db, text):
    text = str(text or "")
    found = {}
    persons = db.query(Person).filter_by(status="active").all()
    for person in persons:
        if person.person_key == "self":
            continue
        names = [str(person.canonical_name).strip()] if person.canonical_name else []
        aliases = db.query(PersonAlias).filter_by(person_id=person.id).all()
        names.extend([str(a.alias).strip() for a in aliases if a.alias])
        for name in names:
            if name and name in text:
                found[person.person_key] = person
                break
    return found

