from models import Person, PersonAlias, Identity, Conversation, Memory, ItemStatus, Source
from services.person_service import (
    cleanup_self_duplicates,
    ensure_alias,
    ensure_identity,
    merge_person_into_self,
    get_person_by_alias,
    get_person_by_identity,
    make_person_key,
)

def command_name_list(db):
    persons = db.query(Person).filter(Person.status.in_(["active"])).order_by(Person.id.asc()).all()
    if not persons:
        return "아직 아는 사람이 없어"
    lines = ["[아는 인간 목록]"]
    for p in persons:
        aliases = list(dict.fromkeys([a.alias for a in p.aliases if a.alias]))
        identities = list(dict.fromkeys([i.display_name for i in db.query(Identity).filter_by(person_id=p.id, platform="kakaotalk").all() if i.display_name]))
        p_type = "본인" if p.person_key == "self" else "타인"
        obs = "채팅에서 봄" if p.observed_in_chat else "채팅에서 아직 못 봄"
        lines.append(f"{p.person_key} | {p.canonical_name} | {p_type} | {obs} | 별칭: {', '.join(aliases) or '-'} | 카톡ID: {', '.join(identities) or '-'}")
    return "\n".join(lines)

def command_name_delete(db, name):
    name = str(name or "").strip()
    if not name: return "삭제할 이름을 적어줘"
    self_person = db.query(Person).filter_by(person_key="self").first()
    deleted = False

    for alias in db.query(PersonAlias).filter_by(alias=name).all():
        db.delete(alias)
        deleted = True
    for identity in db.query(Identity).filter_by(display_name=name, platform="kakaotalk").all():
        db.delete(identity)
        deleted = True
    db.commit()

    if not deleted: return f"{name}이라는 이름은 없어"
    return f"{name} 이름 연결만 삭제햇어 (대화/기억은 그대로임)"

def command_name(db, sender, new_name, persona_id):
    sender = str(sender or "").strip()
    new_name = str(new_name or "").strip().lstrip("@")
    if not sender: return "sender가 없어"
    if not new_name: return "이름을 적어줘"

    self_person = cleanup_self_duplicates(db, sender, new_name, persona_id)
    self_person.canonical_name = new_name
    self_person.person_type = "self"
    self_person.status = "active"
    self_person.confirmed = 1
    self_person.observed_in_chat = 1

    identities = db.query(Identity).filter_by(display_name=sender, platform="kakaotalk").all()
    for identity in identities:
        identity.person_id = self_person.id
        identity.target_key = "self"
        identity.is_primary = 1
    if not identities:
        ensure_identity(db, self_person, sender, is_primary=1)

    for n_id in db.query(Identity).filter_by(display_name=new_name, platform="kakaotalk").all():
        n_id.person_id = self_person.id
        n_id.target_key = "self"
        n_id.is_primary = 1

    for name_to_alias in [sender, new_name]:
        s_alias = db.query(PersonAlias).filter_by(alias=name_to_alias).first()
        if s_alias and s_alias.person_id != self_person.id and s_alias.person:
            merge_person_into_self(db, s_alias.person, self_person, persona_id)
        ensure_alias(db, self_person, name_to_alias)

    db.commit()
    return f"{sender} -> {new_name} (self) 연결햇어"

def command_person(db, canonical_name, aliases):
    canonical_name = str(canonical_name or "").strip()
    if not canonical_name: return "인물 이름을 적어줘"
    self_p = db.query(Person).filter_by(person_key="self").first()
    if self_p and canonical_name == self_p.canonical_name:
        return f"{canonical_name}은 이미 본인(self)으로 등록돼있어"

    person = get_person_by_alias(db, canonical_name) or get_person_by_identity(db, canonical_name)
    if not person:
        person = Person(person_key=make_person_key(db), canonical_name=canonical_name, person_type="person", status="active", confirmed=1)
        db.add(person)
        db.commit()
        db.refresh(person)

    for name in [canonical_name] + aliases:
        name = str(name or "").strip()
        if not name: continue
        ex = db.query(PersonAlias).filter_by(alias=name).first()
        if ex:
            if ex.person_id != person.id:
                return f"{name}은 이미 {ex.person.canonical_name}으로 등록돼있어"
            continue
        ensure_alias(db, person, name)

    person.confirmed = 1
    person.status = "active"
    db.commit()
    return f"{person.canonical_name} 등록햇어 ({person.person_key})"

def command_person_delete(db, name):
    name = str(name or "").strip()
    person = get_person_by_alias(db, name)
    if not person: return f"{name}이라는 인물을 못 찾겠어"
    if person.person_key == "self": return "본인은 인물삭제 말고 /이름삭제를 써"
    person.status = "inactive"
    for identity in db.query(Identity).filter_by(person_id=person.id).all():
        identity.is_primary = 0
    db.commit()
    return f"{person.canonical_name} 비활성화햇어 ({person.person_key})"

def command_person_merge(db, old_name, target_name, persona_id):
    old_p = get_person_by_alias(db, old_name)
    tgt_p = get_person_by_alias(db, target_name)
    if not old_p: return f"{old_name}을 못 찾겠어"
    if not tgt_p: return f"{target_name}을 못 찾겠어"
    if old_p.id == tgt_p.id: return "이미 같은 사람이야"

    old_key, target_key = old_p.person_key, tgt_p.person_key
    for a in db.query(PersonAlias).filter_by(person_id=old_p.id).all():
        if db.query(PersonAlias).filter(PersonAlias.alias == a.alias, PersonAlias.person_id == tgt_p.id).first():
            db.delete(a)
        else:
            a.person_id = tgt_p.id

    for i in db.query(Identity).filter_by(person_id=old_p.id).all():
        if db.query(Identity).filter(Identity.display_name == i.display_name, Identity.person_id == tgt_p.id).first():
            db.delete(i)
        else:
            i.person_id = tgt_p.id
            i.target_key = target_key

    for c in db.query(Conversation).filter(Conversation.speaker_id == old_key).all():
        c.speaker_id = target_key

    for m in db.query(Memory).filter(Memory.persona_id == persona_id).all():
        if old_key in (m.people_involved or []):
            m.people_involved = list(dict.fromkeys([target_key if v == old_key else v for v in m.people_involved]))

    old_p.status = "merged"
    tgt_p.confirmed = 1
    db.commit()
    return f"{old_name} -> {target_name} 병합햇어"

def handle_identity_commands(db, sender, user_input, persona_id):
    if user_input in ["/이름목록", "/이름 목록", "/인물목록", "/인물 목록"]:
        return command_name_list(db)
    if user_input.startswith("/이름삭제 "):
        return command_name_delete(db, user_input[6:].strip())
    if user_input.startswith("/이름 "):
        return command_name(db, sender, user_input[4:].strip(), persona_id)
    if user_input.startswith("/인물삭제 "):
        return command_person_delete(db, user_input[6:].strip())
    if user_input.startswith("/인물병합 "):
        args = user_input[6:].split()
        return command_person_merge(db, args[0], args[1], persona_id) if len(args) >= 2 else "합칠 두 이름을 적어줘"
    if user_input.startswith("/인물 "):
        args = user_input[4:].split()
        return command_person(db, args[0], args[1:]) if args else "인물 이름을 적어줘"
    return None

